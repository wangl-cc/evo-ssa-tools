//! Replacing-manifest commit execution and publication.

use std::{cmp::Ordering, fs, mem, ops::Range, sync::Arc};

use super::{
    Committer,
    batch::{PreparedBatch, WriteBatch},
    options::CommitOptions,
    plan::{CommitPlan, CommitPublicationStats, StagedSegment},
};
use crate::{
    catalog::{SegmentManifestEntry, SegmentTier},
    error::{InputError, Result},
    record::EntrySource,
    segment::{Segment, SegmentWriter},
    snapshot::RangeCursor,
};

/// Summary returned after a successful batch commit.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct CommitStats {
    /// Records accepted from the caller's batch.
    pub input_records: usize,
    /// Sum of key and value bytes accepted from the caller.
    pub input_bytes: usize,
    /// Segment files made visible by the manifest update.
    pub segments_published: usize,
    /// Manifest entries removed by this publication: the rebuilt run that the
    /// batch normalized with, plus any dead entries dropped along the way.
    pub segments_retired: usize,
    /// Logical records written into newly published segments. Equals
    /// `input_records` for direct main and patch publishes; exceeds it when the
    /// commit normalizes patch/main overlap into replacement main segments.
    pub output_records: usize,
}

/// Result of asking a sorted record stream to append its next record.
pub(super) enum RecordStreamStep {
    Pushed,
    ChunkFull,
    Done,
}

/// Sorted unique records that can pause before a record that exceeds a chunk.
pub(super) trait SortedRecordStream {
    fn push_next_into(&mut self, chunk: &mut SegmentChunk) -> Result<RecordStreamStep>;
}

/// One segment-sized output batch with exact record and byte boundaries.
///
/// A single record larger than the byte threshold is accepted into an empty
/// chunk so every input record can make progress. Otherwise both configured
/// limits are hard upper bounds.
pub(super) struct SegmentChunk {
    batch: WriteBatch,
    max_records: usize,
    max_bytes: usize,
}

impl SegmentChunk {
    pub(super) fn new(max_records: usize, max_bytes: usize) -> Self {
        debug_assert!(max_records > 0);
        debug_assert!(max_bytes > 0);
        Self {
            batch: WriteBatch::new(),
            max_records,
            max_bytes,
        }
    }

    pub(super) fn try_push(&mut self, key: &[u8], value: &[u8]) -> bool {
        if !self.batch.is_empty() {
            let record_bytes = key.len().saturating_add(value.len());
            let exceeds_records = self.batch.len() >= self.max_records;
            let exceeds_bytes = self.batch.byte_len() > self.max_bytes.saturating_sub(record_bytes);
            if exceeds_records || exceeds_bytes {
                return false;
            }
        }
        self.batch.push(key, value);
        true
    }

    pub(super) fn is_empty(&self) -> bool {
        self.batch.is_empty()
    }

    pub(super) fn take(&mut self) -> WriteBatch {
        mem::take(&mut self.batch)
    }
}

struct PendingSegment<M> {
    segment_id: u32,
    tier: SegmentTier,
    metadata: M,
}

struct NormalizationRecords<'a> {
    batch: &'a PreparedBatch,
    batch_index: usize,
    existing: RangeCursor,
}

impl<'a> NormalizationRecords<'a> {
    fn new(batch: &'a PreparedBatch, existing: RangeCursor) -> Self {
        Self {
            batch,
            batch_index: 0,
            existing,
        }
    }
}

impl SortedRecordStream for NormalizationRecords<'_> {
    fn push_next_into(&mut self, chunk: &mut SegmentChunk) -> Result<RecordStreamStep> {
        enum Advance {
            Batch,
            Existing,
            Both,
            Done,
        }

        let advance = {
            let existing_record = self.existing.current_record()?;
            match (self.batch_index < self.batch.len(), existing_record) {
                (true, Some(existing_record)) => {
                    let entry = self.batch.entry(self.batch_index);
                    match entry.key().cmp(existing_record.key) {
                        Ordering::Less => {
                            if !chunk.try_push(entry.key(), entry.value()) {
                                return Ok(RecordStreamStep::ChunkFull);
                            }
                            Advance::Batch
                        }
                        Ordering::Greater => {
                            if !chunk.try_push(existing_record.key, existing_record.value) {
                                return Ok(RecordStreamStep::ChunkFull);
                            }
                            Advance::Existing
                        }
                        Ordering::Equal => {
                            let winner = if entry.value() <= existing_record.value {
                                entry.value()
                            } else {
                                existing_record.value
                            };
                            if !chunk.try_push(entry.key(), winner) {
                                return Ok(RecordStreamStep::ChunkFull);
                            }
                            Advance::Both
                        }
                    }
                }
                (true, None) => {
                    let entry = self.batch.entry(self.batch_index);
                    if !chunk.try_push(entry.key(), entry.value()) {
                        return Ok(RecordStreamStep::ChunkFull);
                    }
                    Advance::Batch
                }
                (false, Some(existing_record)) => {
                    if !chunk.try_push(existing_record.key, existing_record.value) {
                        return Ok(RecordStreamStep::ChunkFull);
                    }
                    Advance::Existing
                }
                (false, None) => Advance::Done,
            }
        };

        match advance {
            Advance::Batch => self.batch_index += 1,
            Advance::Existing => self.existing.advance_record()?,
            Advance::Both => {
                self.batch_index += 1;
                self.existing.advance_record()?;
            }
            Advance::Done => return Ok(RecordStreamStep::Done),
        }
        Ok(RecordStreamStep::Pushed)
    }
}

impl Committer<'_> {
    /// Runs one batch commit. The public entry points `Store::commit_batch` and
    /// `Store::commit_batch_with_options` live on the store facade and delegate
    /// here; this is the whole replacing-manifest publication algorithm.
    pub(crate) fn commit_with_options(
        &self,
        batch: WriteBatch,
        options: &CommitOptions,
    ) -> Result<CommitStats> {
        if self.inner.writer_lock.is_none() {
            return Err(InputError::ReadOnlyStore.into());
        }
        if batch.is_empty() {
            return Ok(CommitStats::default());
        }
        let geometry = self.inner.geometry;
        let key_len = geometry.key_len;
        let batch = batch.prepare_for(key_len, geometry.value_layout)?;
        let input_records = batch.len();
        let input_bytes = batch.byte_len();

        let _commit_guard = self.inner.commit_lock.lock();

        let (manifest, main_segments, patch_segments) = {
            let state = self.inner.state.read();
            (
                state.manifest.clone(),
                state.main_segments_as_vec(),
                state.patch_segments_as_vec(),
            )
        };

        let batch_min = batch.key_at(0).to_vec();
        let batch_max = batch.key_at(input_records - 1).to_vec();
        let mut plan = CommitPlan::from_snapshot(manifest, main_segments, patch_segments);
        let affected_main = plan.affected_main_range(&batch_min, &batch_max);
        let direct_ranges = batch.flush_ranges(
            key_len,
            options.flush_threshold_records(),
            options.flush_threshold_bytes(),
        );

        let (written, output_records) = if affected_main.is_empty() {
            let written = self.write_batch_segments(
                &batch,
                direct_ranges,
                SegmentTier::Main,
                &mut plan,
                options,
            )?;
            (written, input_records)
        } else if plan.should_publish_patch(input_records, direct_ranges.len(), options) {
            let written = self.write_batch_segments(
                &batch,
                direct_ranges,
                SegmentTier::Patch,
                &mut plan,
                options,
            )?;
            (written, input_records)
        } else {
            let (written, merged_len) = self.write_normalized_segments(
                &mut plan,
                &batch,
                Some((batch_min.as_slice(), batch_max.as_slice())),
                options,
            )?;
            (written, merged_len)
        };

        let publication_stats = self.publish_plan(plan, written, key_len)?;

        Ok(CommitStats {
            input_records,
            input_bytes,
            segments_published: publication_stats.segments_published,
            segments_retired: publication_stats.segments_retired,
            output_records,
        })
    }

    /// Normalizes every live patch segment into the main tier.
    pub(crate) fn normalize_patches_with_options(
        &self,
        options: &CommitOptions,
    ) -> Result<CommitStats> {
        if self.inner.writer_lock.is_none() {
            return Err(InputError::ReadOnlyStore.into());
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
        if !plan.has_patches() && !plan.has_dead_entries() {
            return Ok(CommitStats::default());
        }

        let (written, output_records) =
            self.write_normalized_segments(&mut plan, &PreparedBatch::default(), None, options)?;
        let publication_stats = self.publish_plan(plan, written, key_len)?;
        Ok(CommitStats {
            input_records: 0,
            input_bytes: 0,
            segments_published: publication_stats.segments_published,
            segments_retired: publication_stats.segments_retired,
            output_records,
        })
    }

    fn write_normalized_segments(
        &self,
        plan: &mut CommitPlan,
        batch: &PreparedBatch,
        batch_bounds: Option<(&[u8], &[u8])>,
        options: &CommitOptions,
    ) -> Result<(Vec<StagedSegment>, usize)> {
        let affected_live = if let Some((batch_min, batch_max)) = batch_bounds {
            plan.retire_normalized_segments(batch_min, batch_max)
        } else {
            let Some(affected_live) = plan.retire_patch_normalization_segments() else {
                return Ok((Vec::new(), 0));
            };
            affected_live
        };

        let existing = RangeCursor::from_segment_sets(
            affected_live.main,
            affected_live.patches,
            self.inner.geometry,
            self.inner.verify_block_checksums,
            None,
            None,
        );
        let mut records = NormalizationRecords::new(batch, existing);
        self.write_record_segments(&mut records, plan, options)
    }

    pub(super) fn publish_plan(
        &self,
        plan: CommitPlan,
        written: Vec<StagedSegment>,
        key_len: usize,
    ) -> Result<CommitPublicationStats> {
        let publication = plan.into_publication(written, key_len)?;
        self.inner.paths.publish_manifest(&publication.manifest)?;

        let stats = CommitPublicationStats {
            segments_published: publication.segments_published,
            segments_retired: publication.segments_retired,
        };

        {
            let mut state = self.inner.state.write();
            state.replace_segments(publication.main_runtime, publication.patch_runtime);
            state.manifest = publication.manifest;
        }

        Ok(stats)
    }

    /// Streams sorted unique records into bounded staged segments.
    pub(super) fn write_record_segments<R: SortedRecordStream>(
        &self,
        records: &mut R,
        plan: &mut CommitPlan,
        options: &CommitOptions,
    ) -> Result<(Vec<StagedSegment>, usize)> {
        let geometry = self.inner.geometry;
        let writer = self.segment_writer(options);
        let mut segment_batch = self.inner.paths.segment_batch();
        let mut pending = Vec::new();
        let mut chunk = SegmentChunk::new(
            options.flush_threshold_records(),
            options.flush_threshold_bytes(),
        );
        let mut output_records = 0usize;

        {
            let mut stage_chunk = |chunk: &mut SegmentChunk| -> Result<()> {
                if chunk.is_empty() {
                    return Ok(());
                }
                let batch = PreparedBatch::from_sorted_unique(
                    chunk.take(),
                    geometry.key_len,
                    geometry.value_layout,
                );
                let segment_id = plan.allocate_segment_id()?;
                let entries = batch.view(0..batch.len());
                let metadata = segment_batch
                    .stage_with(segment_id, |file| Ok(writer.write(file, &entries)?))?;
                pending.push(PendingSegment {
                    segment_id,
                    tier: SegmentTier::Main,
                    metadata,
                });
                Ok(())
            };

            loop {
                match records.push_next_into(&mut chunk)? {
                    RecordStreamStep::Pushed => output_records += 1,
                    RecordStreamStep::ChunkFull => stage_chunk(&mut chunk)?,
                    RecordStreamStep::Done => break,
                }
            }
            stage_chunk(&mut chunk)?;
        }
        segment_batch.publish()?;

        let mut written = Vec::with_capacity(pending.len());
        for PendingSegment {
            segment_id,
            tier,
            metadata,
        } in pending
        {
            let segment_len = metadata.segment_len();
            let content_id = metadata.content_id();
            let file = fs::File::open(self.inner.paths.final_segment(segment_id))?;
            let runtime = Arc::new(Segment::from_written(segment_id, file, metadata));
            let entry = SegmentManifestEntry::new(
                segment_id,
                tier,
                segment_len,
                content_id,
                runtime.min_key().to_vec(),
                runtime.max_key().to_vec(),
            );
            written.push(StagedSegment::new(entry, runtime));
        }
        Ok((written, output_records))
    }

    /// Stages and publishes segment files for contiguous prepared-batch ranges.
    fn write_batch_segments(
        &self,
        batch: &PreparedBatch,
        ranges: Vec<Range<usize>>,
        tier: SegmentTier,
        plan: &mut CommitPlan,
        options: &CommitOptions,
    ) -> Result<Vec<StagedSegment>> {
        let writer = self.segment_writer(options);
        let mut segment_batch = self.inner.paths.segment_batch();
        let mut pending = Vec::with_capacity(ranges.len());
        for range in ranges {
            let segment_id = plan.allocate_segment_id()?;
            let entries = batch.view(range);
            let metadata =
                segment_batch.stage_with(segment_id, |file| Ok(writer.write(file, &entries)?))?;
            pending.push(PendingSegment {
                segment_id,
                tier,
                metadata,
            });
        }
        segment_batch.publish()?;

        let mut written = Vec::with_capacity(pending.len());
        for PendingSegment {
            segment_id,
            tier,
            metadata,
        } in pending
        {
            let segment_len = metadata.segment_len();
            let content_id = metadata.content_id();
            let file = fs::File::open(self.inner.paths.final_segment(segment_id))?;
            let runtime = Arc::new(Segment::from_written(segment_id, file, metadata));
            let entry = SegmentManifestEntry::new(
                segment_id,
                tier,
                segment_len,
                content_id,
                runtime.min_key().to_vec(),
                runtime.max_key().to_vec(),
            );
            written.push(StagedSegment::new(entry, runtime));
        }
        Ok(written)
    }

    #[cfg(feature = "value-compression")]
    fn segment_writer(&self, options: &CommitOptions) -> SegmentWriter {
        let geometry = self.inner.geometry;
        SegmentWriter::new(
            geometry,
            options.value_payload_compression_policy(),
            options.target_block_size(),
        )
    }

    #[cfg(not(feature = "value-compression"))]
    fn segment_writer(&self, options: &CommitOptions) -> SegmentWriter {
        let geometry = self.inner.geometry;
        SegmentWriter::new(geometry, options.target_block_size())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_chunk_stops_at_record_limit() {
        let mut chunk = SegmentChunk::new(2, usize::MAX);

        assert!(chunk.try_push(b"aa", b"123"));
        assert!(chunk.try_push(b"bb", b"456"));
        assert!(!chunk.try_push(b"cc", b"7"));

        let batch = chunk.take();
        assert_eq!(batch.len(), 2);
        assert_eq!(batch.byte_len(), 10);
        assert!(chunk.is_empty());
    }

    #[test]
    fn segment_chunk_stops_at_exact_byte_limit() {
        let mut chunk = SegmentChunk::new(3, 10);

        assert!(chunk.try_push(b"aa", b"123"));
        assert!(chunk.try_push(b"bb", b"456"));
        assert!(!chunk.try_push(b"cc", b"7"));

        let batch = chunk.take();
        assert_eq!(batch.len(), 2);
        assert_eq!(batch.byte_len(), 10);
    }

    #[test]
    fn segment_chunk_allows_one_record_larger_than_byte_limit() {
        let mut chunk = SegmentChunk::new(2, 3);

        assert!(chunk.try_push(b"key", b"value"));
        assert!(!chunk.try_push(b"next", b"value"));

        let batch = chunk.take();
        assert_eq!(batch.len(), 1);
        assert_eq!(batch.byte_len(), 8);
    }
}
