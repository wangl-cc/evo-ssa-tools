//! Cross-store merge publication.

use std::{mem, sync::Arc};

use crate::{
    catalog::manifest::SegmentTier,
    commit::{
        batch::{PreparedBatch, WriteBatch},
        execution::{CommitPlan, CommitStats, WrittenSegment},
        options::CommitOptions,
    },
    error::{FormatError, InputError, Result},
    schema::StoreGeometry,
    segment::state::SegmentState,
    snapshot::RangeCursor,
    store::Store,
};

struct MergeSourceSnapshot {
    main_segments: Vec<Arc<SegmentState>>,
    patch_segments: Vec<Arc<SegmentState>>,
    min_key: Vec<u8>,
    max_key: Vec<u8>,
}

struct StoreMergeRecords {
    destination: RangeCursor,
    source: RangeCursor,
    source_stats: MergeInputStats,
}

struct MergeInputStats {
    records: usize,
    bytes: usize,
}

impl MergeSourceSnapshot {
    fn from_store(store: &Store) -> Option<Self> {
        let state = store.inner.state.read();
        let (min_key, max_key) = segment_bounds(
            state
                .main_segments
                .iter()
                .chain(state.patch_segments.iter()),
        )?;
        Some(Self {
            main_segments: state.main_segments.iter().cloned().collect(),
            patch_segments: state.patch_segments.iter().cloned().collect(),
            min_key,
            max_key,
        })
    }

    fn cursor(&self, geometry: StoreGeometry, verify_block_checksums: bool) -> RangeCursor {
        RangeCursor::from_segment_sets(
            self.main_segments.clone(),
            self.patch_segments.clone(),
            geometry,
            verify_block_checksums,
            None,
            None,
        )
    }
}

impl MergeInputStats {
    fn new() -> Self {
        Self {
            records: 0,
            bytes: 0,
        }
    }

    fn add_record(&mut self, key: &[u8], value: &[u8]) -> Result<()> {
        self.records = self
            .records
            .checked_add(1)
            .ok_or(FormatError::limit("merge record count"))?;
        let record_bytes = key
            .len()
            .checked_add(value.len())
            .ok_or(FormatError::limit("merge byte length"))?;
        self.bytes = self
            .bytes
            .checked_add(record_bytes)
            .ok_or(FormatError::limit("merge byte length"))?;
        Ok(())
    }
}

impl StoreMergeRecords {
    fn new(destination: RangeCursor, source: RangeCursor) -> Self {
        Self {
            destination,
            source,
            source_stats: MergeInputStats::new(),
        }
    }

    fn into_source_stats(self) -> MergeInputStats {
        self.source_stats
    }

    fn push_next_into(&mut self, batch: &mut WriteBatch) -> Result<bool> {
        enum Advance {
            Destination,
            Source,
            Both,
            Done,
        }

        let advance = {
            let Self {
                destination,
                source,
                source_stats,
            } = self;
            let destination_record = destination.current_record()?;
            let source_record = source.current_record()?;
            match (destination_record, source_record) {
                (Some(destination_record), Some(source_record)) => {
                    match destination_record.key.cmp(source_record.key) {
                        std::cmp::Ordering::Less => {
                            batch.push(destination_record.key, destination_record.value);
                            Advance::Destination
                        }
                        std::cmp::Ordering::Greater => {
                            source_stats.add_record(source_record.key, source_record.value)?;
                            batch.push(source_record.key, source_record.value);
                            Advance::Source
                        }
                        std::cmp::Ordering::Equal => {
                            source_stats.add_record(source_record.key, source_record.value)?;
                            if source_record.value < destination_record.value {
                                batch.push(source_record.key, source_record.value);
                            } else {
                                batch.push(destination_record.key, destination_record.value);
                            }
                            Advance::Both
                        }
                    }
                }
                (Some(destination_record), None) => {
                    batch.push(destination_record.key, destination_record.value);
                    Advance::Destination
                }
                (None, Some(source_record)) => {
                    source_stats.add_record(source_record.key, source_record.value)?;
                    batch.push(source_record.key, source_record.value);
                    Advance::Source
                }
                (None, None) => Advance::Done,
            }
        };

        match advance {
            Advance::Destination => self.destination.advance_record()?,
            Advance::Source => self.source.advance_record()?,
            Advance::Both => {
                self.destination.advance_record()?;
                self.source.advance_record()?;
            }
            Advance::Done => return Ok(false),
        }
        Ok(true)
    }
}

impl Store {
    /// Atomically merges every visible record from `source` into this store.
    pub(crate) fn merge_store_with_options(
        &self,
        source: &Store,
        options: &CommitOptions,
    ) -> Result<CommitStats> {
        if self.inner.writer_lock.is_none() {
            return Err(InputError::ReadOnlyStore.into());
        }
        self.validate_merge_source(source)?;
        if Arc::ptr_eq(&self.inner, &source.inner) {
            return Ok(CommitStats::default());
        }

        let Some(source_snapshot) = MergeSourceSnapshot::from_store(source) else {
            return Ok(CommitStats::default());
        };
        let mut source_cursor = source_snapshot.cursor(source.inner.geometry, true);
        if source_cursor.current_record()?.is_none() {
            return Ok(CommitStats::default());
        }

        let _commit_guard = self.inner.commit_lock.lock();
        let geometry = self.inner.geometry;
        let key_len = geometry.key_len;
        let (manifest, main_segments, patch_segments) = {
            let state = self.inner.state.read();
            (
                state.manifest.clone(),
                state.main_segments_as_vec(),
                state.patch_segments_as_vec(),
            )
        };
        let mut plan = CommitPlan::from_snapshot(manifest, main_segments, patch_segments);
        let affected_live =
            plan.retire_normalized_segments(&source_snapshot.min_key, &source_snapshot.max_key);

        let destination_cursor = RangeCursor::from_segment_sets(
            affected_live.main,
            affected_live.patches,
            geometry,
            self.inner.verify_block_checksums,
            None,
            None,
        );
        let mut records = StoreMergeRecords::new(destination_cursor, source_cursor);
        let (written, output_records) =
            self.write_cursor_segments(&mut records, &mut plan, options)?;
        let source_stats = records.into_source_stats();
        let publication_stats = self.publish_plan(plan, written, key_len)?;

        Ok(CommitStats {
            input_records: source_stats.records,
            input_bytes: source_stats.bytes,
            segments_published: publication_stats.segments_published,
            segments_retired: publication_stats.segments_retired,
            output_records,
        })
    }

    fn validate_merge_source(&self, source: &Store) -> Result<()> {
        if self.inner.metadata != source.inner.metadata {
            return Err(InputError::SourceMetadataMismatch.into());
        }
        if self.inner.geometry.key_len != source.inner.geometry.key_len {
            return Err(InputError::SourceKeyLengthMismatch {
                expected: self.inner.geometry.key_len,
                actual: source.inner.geometry.key_len,
            }
            .into());
        }
        if self.inner.geometry.value_layout != source.inner.geometry.value_layout {
            return Err(InputError::SourceValueLayoutMismatch.into());
        }
        Ok(())
    }

    fn write_cursor_segments(
        &self,
        records: &mut StoreMergeRecords,
        plan: &mut CommitPlan,
        options: &CommitOptions,
    ) -> Result<(Vec<WrittenSegment>, usize)> {
        let mut written = Vec::new();
        let mut output_records = 0usize;
        let mut batch = WriteBatch::new();
        while records.push_next_into(&mut batch)? {
            output_records += 1;
            if batch.len() >= options.flush_threshold_records()
                || batch.byte_len() >= options.flush_threshold_bytes()
            {
                self.flush_cursor_batch(&mut batch, &mut written, plan, options)?;
            }
        }
        self.flush_cursor_batch(&mut batch, &mut written, plan, options)?;
        Ok((written, output_records))
    }

    fn flush_cursor_batch(
        &self,
        batch: &mut WriteBatch,
        written: &mut Vec<WrittenSegment>,
        plan: &mut CommitPlan,
        options: &CommitOptions,
    ) -> Result<()> {
        if batch.is_empty() {
            return Ok(());
        }
        let geometry = self.inner.geometry;
        let batch = PreparedBatch::from_sorted_unique(
            mem::take(batch),
            geometry.key_len,
            geometry.value_layout,
        );
        let len = batch.len();
        let segment_id = plan.allocate_segment_id()?;
        written.push(self.write_segment(&batch, 0..len, segment_id, SegmentTier::Main, options)?);
        Ok(())
    }
}

fn segment_bounds<'a>(
    mut segments: impl Iterator<Item = &'a Arc<SegmentState>>,
) -> Option<(Vec<u8>, Vec<u8>)> {
    let first = segments.next()?;
    let mut min_key = first.min_key.clone();
    let mut max_key = first.max_key.clone();
    for segment in segments {
        if segment.min_key < min_key {
            min_key = segment.min_key.clone();
        }
        if segment.max_key > max_key {
            max_key = segment.max_key.clone();
        }
    }
    Some((min_key, max_key))
}
