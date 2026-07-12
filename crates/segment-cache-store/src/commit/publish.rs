//! Replacing-manifest commit execution and publication.

use std::{cmp::Ordering, fs, ops::Range, sync::Arc};

use super::{
    Committer,
    batch::{PreparedBatch, WriteBatch},
    options::CommitOptions,
    plan::{CommitPlan, CommitPublicationStats, RetiredSegments, StagedSegment},
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

        let merged = self.merge_region(batch, affected_live)?;
        let ranges = merged.flush_ranges(
            self.inner.geometry.key_len,
            options.flush_threshold_records(),
            options.flush_threshold_bytes(),
        );
        let written =
            self.write_batch_segments(&merged, ranges, SegmentTier::Main, plan, options)?;
        Ok((written, merged.len()))
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

    /// Merges `batch` with the records of the intersecting `affected` segments
    /// into one sorted, deduplicated batch.
    ///
    /// Duplicate keys (a key present in both the batch and an existing segment)
    /// are resolved by the winner rule: the copy with the lexicographically
    /// smallest value bytes survives. This rule is history-independent, so
    /// replicas that hold the same key set converge.
    fn merge_region(
        &self,
        batch: &PreparedBatch,
        affected: RetiredSegments,
    ) -> Result<PreparedBatch> {
        let geometry = self.inner.geometry;
        let verify = self.inner.verify_block_checksums;
        let mut existing = RangeCursor::from_segment_sets(
            affected.main,
            affected.patches,
            geometry,
            verify,
            None,
            None,
        );

        let mut merged = WriteBatch::new();
        let mut index = 0usize;
        let len = batch.len();
        enum Advance {
            Batch,
            Existing,
            Both,
            Done,
        }

        loop {
            let advance = {
                let existing_record = existing.current_record()?;
                match (index < len, existing_record) {
                    (true, Some(existing_record)) => {
                        let entry = batch.entry(index);
                        match entry.key().cmp(existing_record.key) {
                            Ordering::Less => {
                                merged.push(entry.key(), entry.value());
                                Advance::Batch
                            }
                            Ordering::Greater => {
                                merged.push(existing_record.key, existing_record.value);
                                Advance::Existing
                            }
                            Ordering::Equal => {
                                let winner = if entry.value() <= existing_record.value {
                                    entry.value()
                                } else {
                                    existing_record.value
                                };
                                merged.push(entry.key(), winner);
                                Advance::Both
                            }
                        }
                    }
                    (true, None) => {
                        let entry = batch.entry(index);
                        merged.push(entry.key(), entry.value());
                        Advance::Batch
                    }
                    (false, Some(existing_record)) => {
                        merged.push(existing_record.key, existing_record.value);
                        Advance::Existing
                    }
                    (false, None) => Advance::Done,
                }
            };

            match advance {
                Advance::Batch => {
                    index += 1;
                }
                Advance::Existing => {
                    existing.advance_record()?;
                }
                Advance::Both => {
                    index += 1;
                    existing.advance_record()?;
                }
                Advance::Done => break,
            }
        }
        Ok(PreparedBatch::from_sorted_unique(
            merged,
            geometry.key_len,
            geometry.value_layout,
        ))
    }

    /// Writes one segment file from a contiguous range of a merged batch.
    fn write_batch_segments(
        &self,
        batch: &PreparedBatch,
        ranges: Vec<Range<usize>>,
        tier: SegmentTier,
        plan: &mut CommitPlan,
        options: &CommitOptions,
    ) -> Result<Vec<StagedSegment>> {
        let mut written = Vec::with_capacity(ranges.len());
        for range in ranges {
            let segment_id = plan.allocate_segment_id()?;
            written.push(self.write_segment(batch, range, segment_id, tier, options)?);
        }
        Ok(written)
    }

    /// Writes one segment file from a contiguous range of a logical batch.
    pub(super) fn write_segment(
        &self,
        batch: &PreparedBatch,
        range: Range<usize>,
        segment_id: u32,
        tier: SegmentTier,
        options: &CommitOptions,
    ) -> Result<StagedSegment> {
        let segment_entries = batch.view(range);
        let segment_paths = self.inner.paths.segment_publish_path(segment_id);
        let writer = self.segment_writer(options);
        let metadata = segment_paths
            .publish()
            .write_with(|file| Ok(writer.write(file, &segment_entries)?))?;
        let fingerprint = metadata.fingerprint();
        let file = fs::File::open(segment_paths.final_path())?;
        let runtime = Arc::new(Segment::from_written(segment_id, file, metadata));
        let min_key = runtime.min_key().to_vec();
        let max_key = runtime.max_key().to_vec();
        let entry = SegmentManifestEntry::new(segment_id, tier, fingerprint, min_key, max_key);
        Ok(StagedSegment::new(entry, runtime))
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
