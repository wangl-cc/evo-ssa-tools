//! Cross-store merge publication.

use std::{mem, sync::Arc};

use crate::{
    engine::runtime::{SegmentState, StoreGeometry},
    error::{InputError, Result},
    format::{FormatError, manifest::SegmentTier},
    read::cursor::{RangeCursor, SegmentRangeCursor},
    store::Store,
    write::{
        WriteBatch,
        commit::{CommitOptions, CommitPlan, CommitStats, WrittenSegment},
    },
};

struct MergeSourceSnapshot {
    segments: Vec<Arc<SegmentState>>,
    has_patches: bool,
    min_key: Vec<u8>,
    max_key: Vec<u8>,
}

struct RecordCursor {
    cursor: RangeCursor,
    pending: Option<(Vec<u8>, Vec<u8>)>,
}

struct SourceRecordCursor {
    inner: RecordCursor,
    stats: MergeInputStats,
}

struct StoreMergeRecords {
    destination: RecordCursor,
    source: SourceRecordCursor,
}

struct MergeInputStats {
    records: usize,
    bytes: usize,
}

impl MergeSourceSnapshot {
    fn from_store(store: &Store) -> Option<Self> {
        let state = store.inner.state.read();
        let has_patches = !state.patch_segments.is_empty();
        let mut segments =
            Vec::with_capacity(state.main_segments.len() + state.patch_segments.len());
        segments.extend(state.main_segments.iter().cloned());
        segments.extend(state.patch_segments.iter().cloned());
        let (min_key, max_key) = segment_bounds(&segments)?;
        Some(Self {
            segments,
            has_patches,
            min_key,
            max_key,
        })
    }

    fn cursor(&self, geometry: StoreGeometry, verify_block_checksums: bool) -> Result<RangeCursor> {
        let mut cursors = Vec::with_capacity(self.segments.len());
        self.push_cursors(&mut cursors, geometry, verify_block_checksums)?;
        if self.has_patches {
            Ok(RangeCursor::merge(cursors))
        } else {
            Ok(RangeCursor::new(cursors))
        }
    }

    fn push_cursors(
        &self,
        cursors: &mut Vec<SegmentRangeCursor>,
        geometry: StoreGeometry,
        verify_block_checksums: bool,
    ) -> Result<()> {
        push_segment_cursors(cursors, &self.segments, geometry, verify_block_checksums)
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

impl RecordCursor {
    fn new(cursor: RangeCursor) -> Result<Self> {
        let mut cursor = cursor;
        let pending = cursor.next().transpose()?;
        Ok(Self { cursor, pending })
    }

    fn is_empty(&self) -> bool {
        self.pending.is_none()
    }

    fn peek(&self) -> Option<(&[u8], &[u8])> {
        self.pending
            .as_ref()
            .map(|(key, value)| (key.as_slice(), value.as_slice()))
    }

    fn take(&mut self) -> Result<Option<(Vec<u8>, Vec<u8>)>> {
        let Some(record) = self.pending.take() else {
            return Ok(None);
        };
        self.pending = self.cursor.next().transpose()?;
        Ok(Some(record))
    }
}

impl SourceRecordCursor {
    fn new(cursor: RangeCursor) -> Result<Self> {
        Ok(Self {
            inner: RecordCursor::new(cursor)?,
            stats: MergeInputStats::new(),
        })
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn peek(&self) -> Option<(&[u8], &[u8])> {
        self.inner.peek()
    }

    fn take(&mut self) -> Result<Option<(Vec<u8>, Vec<u8>)>> {
        let Some(record) = self.inner.take()? else {
            return Ok(None);
        };
        self.stats.add_record(&record.0, &record.1)?;
        Ok(Some(record))
    }

    fn into_stats(self) -> MergeInputStats {
        self.stats
    }
}

impl StoreMergeRecords {
    fn new(destination: RecordCursor, source: SourceRecordCursor) -> Self {
        Self {
            destination,
            source,
        }
    }

    fn into_source_stats(self) -> MergeInputStats {
        self.source.into_stats()
    }
}

impl Iterator for StoreMergeRecords {
    type Item = Result<(Vec<u8>, Vec<u8>)>;

    fn next(&mut self) -> Option<Self::Item> {
        Some(match (self.destination.peek(), self.source.peek()) {
            (Some((destination_key, _)), Some((source_key, _))) => {
                match destination_key.cmp(source_key) {
                    std::cmp::Ordering::Less => self
                        .destination
                        .take()
                        .map(|record| record.expect("destination was pending")),
                    std::cmp::Ordering::Greater => self
                        .source
                        .take()
                        .map(|record| record.expect("source was pending")),
                    std::cmp::Ordering::Equal => self.next_duplicate_key(),
                }
            }
            (Some(_), None) => self
                .destination
                .take()
                .map(|record| record.expect("destination was pending")),
            (None, Some(_)) => self
                .source
                .take()
                .map(|record| record.expect("source was pending")),
            (None, None) => return None,
        })
    }
}

impl StoreMergeRecords {
    fn next_duplicate_key(&mut self) -> Result<(Vec<u8>, Vec<u8>)> {
        let destination = self.destination.take()?.expect("destination was pending");
        let source = self.source.take()?.expect("source was pending");
        if source.1 < destination.1 {
            Ok(source)
        } else {
            Ok(destination)
        }
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
        options.validate()?;
        self.validate_merge_source(source)?;
        if Arc::ptr_eq(&self.inner, &source.inner) {
            return Ok(CommitStats::default());
        }

        let Some(source_snapshot) = MergeSourceSnapshot::from_store(source) else {
            return Ok(CommitStats::default());
        };
        let source_cursor = source_snapshot.cursor(source.inner.geometry, true)?;
        let source_records = SourceRecordCursor::new(source_cursor)?;
        if source_records.is_empty() {
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

        let mut cursors = Vec::with_capacity(affected_live.len());
        push_segment_cursors(
            &mut cursors,
            &affected_live,
            geometry,
            self.inner.verify_block_checksums,
        )?;
        let destination_records = RecordCursor::new(RangeCursor::merge(cursors))?;
        let mut records = StoreMergeRecords::new(destination_records, source_records);
        let (written, merged_records) =
            self.write_cursor_segments(&mut records, &mut plan, options)?;
        let source_stats = records.into_source_stats();
        let publication_stats = self.publish_plan(plan, written, key_len)?;

        Ok(CommitStats {
            records: source_stats.records,
            bytes: source_stats.bytes,
            segments_published: publication_stats.segments_published,
            segments_retired: publication_stats.segments_retired,
            merged_records,
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
        records: impl Iterator<Item = Result<(Vec<u8>, Vec<u8>)>>,
        plan: &mut CommitPlan,
        options: &CommitOptions,
    ) -> Result<(Vec<WrittenSegment>, usize)> {
        let mut written = Vec::new();
        let mut merged_records = 0usize;
        let mut batch = WriteBatch::default();
        for record in records {
            let (key, value) = record?;
            batch.push_owned(key, value)?;
            if batch.len() >= options.flush_threshold_records()
                || batch.bytes >= options.flush_threshold_bytes()
            {
                merged_records +=
                    self.flush_cursor_batch(&mut batch, &mut written, plan, options)?;
            }
        }
        merged_records += self.flush_cursor_batch(&mut batch, &mut written, plan, options)?;
        Ok((written, merged_records))
    }

    fn flush_cursor_batch(
        &self,
        batch: &mut WriteBatch,
        written: &mut Vec<WrittenSegment>,
        plan: &mut CommitPlan,
        options: &CommitOptions,
    ) -> Result<usize> {
        if batch.is_empty() {
            return Ok(0);
        }
        let len = batch.len();
        let batch = mem::take(batch).mark_sorted();
        let segment_id = plan.allocate_segment_id()?;
        written.push(self.write_segment(&batch, 0..len, segment_id, SegmentTier::Main, options)?);
        Ok(len)
    }
}

fn segment_bounds(segments: &[Arc<SegmentState>]) -> Option<(Vec<u8>, Vec<u8>)> {
    let first = segments.first()?;
    let mut min_key = first.min_key.clone();
    let mut max_key = first.max_key.clone();
    for segment in &segments[1..] {
        if segment.min_key < min_key {
            min_key = segment.min_key.clone();
        }
        if segment.max_key > max_key {
            max_key = segment.max_key.clone();
        }
    }
    Some((min_key, max_key))
}

fn push_segment_cursors(
    cursors: &mut Vec<SegmentRangeCursor>,
    segments: &[Arc<SegmentState>],
    geometry: StoreGeometry,
    verify_block_checksums: bool,
) -> Result<()> {
    for segment in segments {
        cursors.push(SegmentRangeCursor::new(
            Arc::clone(segment),
            geometry,
            verify_block_checksums,
            None,
            None,
        )?);
    }
    Ok(())
}
