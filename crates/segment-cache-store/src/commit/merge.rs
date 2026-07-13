//! Cross-store merge publication.

use std::sync::Arc;

use super::{
    Committer,
    options::CommitOptions,
    plan::CommitPlan,
    publish::{CommitStats, RecordStreamStep, SegmentChunk, SortedRecordStream},
};
use crate::{
    error::{FormatError, InputError, Result},
    segment::{Segment, SegmentGeometry},
    snapshot::RangeCursor,
    store::StoreInner,
};

struct MergeSourceSnapshot {
    main_segments: Vec<Arc<Segment>>,
    patch_segments: Vec<Arc<Segment>>,
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
    fn from_store(store: &StoreInner) -> Option<Self> {
        let state = store.state.read();
        if state.main_segments.is_empty() && state.patch_segments.is_empty() {
            return None;
        }
        Some(Self {
            main_segments: state.main_segments.iter().cloned().collect(),
            patch_segments: state.patch_segments.iter().cloned().collect(),
        })
    }

    fn cursor(&self, geometry: SegmentGeometry, verify_block_checksums: bool) -> RangeCursor {
        RangeCursor::strict_from_segment_sets(
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
}

impl SortedRecordStream for StoreMergeRecords {
    fn push_next_into(&mut self, chunk: &mut SegmentChunk) -> Result<RecordStreamStep> {
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
                            if !chunk.try_push(destination_record.key, destination_record.value) {
                                return Ok(RecordStreamStep::ChunkFull);
                            }
                            Advance::Destination
                        }
                        std::cmp::Ordering::Greater => {
                            if !chunk.try_push(source_record.key, source_record.value) {
                                return Ok(RecordStreamStep::ChunkFull);
                            }
                            source_stats.add_record(source_record.key, source_record.value)?;
                            Advance::Source
                        }
                        std::cmp::Ordering::Equal => {
                            if source_record.value != destination_record.value {
                                return Err(InputError::KeyConflict.into());
                            }
                            if !chunk.try_push(source_record.key, source_record.value) {
                                return Ok(RecordStreamStep::ChunkFull);
                            }
                            source_stats.add_record(source_record.key, source_record.value)?;
                            Advance::Both
                        }
                    }
                }
                (Some(destination_record), None) => {
                    if !chunk.try_push(destination_record.key, destination_record.value) {
                        return Ok(RecordStreamStep::ChunkFull);
                    }
                    Advance::Destination
                }
                (None, Some(source_record)) => {
                    if !chunk.try_push(source_record.key, source_record.value) {
                        return Ok(RecordStreamStep::ChunkFull);
                    }
                    source_stats.add_record(source_record.key, source_record.value)?;
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
            Advance::Done => return Ok(RecordStreamStep::Done),
        }
        Ok(RecordStreamStep::Pushed)
    }
}

impl Committer<'_> {
    /// Atomically merges every visible record from `source` into this store.
    pub(crate) fn merge_store_with_options(
        &self,
        source: &Arc<StoreInner>,
        options: &CommitOptions,
    ) -> Result<CommitStats> {
        if self.inner.writer_lock.is_none() {
            return Err(InputError::ReadOnlyStore.into());
        }
        self.validate_merge_source(source)?;
        if Arc::ptr_eq(self.inner, source) {
            return Ok(CommitStats::default());
        }

        let Some(source_snapshot) = MergeSourceSnapshot::from_store(source) else {
            return Ok(CommitStats::default());
        };
        let mut source_cursor = source_snapshot.cursor(source.geometry, true);
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
        let affected_live = plan.retire_all_live_segments();

        let destination_cursor = RangeCursor::strict_from_segment_sets(
            affected_live.main,
            affected_live.patches,
            geometry,
            self.inner.verify_block_checksums,
            None,
            None,
        );
        let mut records = StoreMergeRecords::new(destination_cursor, source_cursor);
        let (written, output_records) =
            self.write_record_segments(&mut records, &mut plan, options)?;
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

    fn validate_merge_source(&self, source: &StoreInner) -> Result<()> {
        if self.inner.metadata != source.metadata {
            return Err(InputError::SourceMetadataMismatch.into());
        }
        if self.inner.geometry.key_len != source.geometry.key_len {
            return Err(InputError::SourceKeyLengthMismatch {
                expected: self.inner.geometry.key_len,
                actual: source.geometry.key_len,
            }
            .into());
        }
        if self.inner.geometry.value_layout != source.geometry.value_layout {
            return Err(InputError::SourceValueLayoutMismatch.into());
        }
        Ok(())
    }
}
