//! Batch commit options and replacing-manifest segment publication.

use std::{
    cmp::Ordering,
    collections::{BTreeSet, HashSet},
    fs,
    ops::Range,
    sync::Arc,
};

use crate::{
    engine::{paths, runtime::SegmentState, segment_file::SegmentFingerprintWriter},
    error::{InputError, OptionsError, Result},
    format::{
        CatalogMismatch, ValuePayloadCompressionPolicy,
        manifest::{
            SegmentManifestEntry, SegmentTier, StoreManifest, validate_segment_entry_shape,
        },
        record::EntrySource,
        segment::SegmentWriter,
    },
    read::cursor::{RangeCursor, SegmentRangeCursor},
    store::Store,
    write::batch::WriteBatch,
};

const DEFAULT_PATCH_SEGMENT_LIMIT: usize = 8;
const DEFAULT_PATCH_DIRECT_RECORD_LIMIT: usize = 4_096;

/// Options consumed by one batch commit.
///
/// These fields affect newly written segment files only. They are not part of
/// the namespace identity and are not required when opening an existing store.
#[derive(Clone, Debug)]
pub struct CommitOptions {
    /// Target logical block split size for newly written segments.
    target_block_size: usize,
    /// Writer-side policy for deciding whether value payloads are worth compressing.
    value_payload_compression_policy: ValuePayloadCompressionPolicy,
    /// Maximum records per newly published segment chunk.
    flush_threshold_records: usize,
    /// Maximum approximate key/value bytes per newly published segment chunk.
    flush_threshold_bytes: usize,
    /// Maximum live patch segments allowed before the next overlapping commit normalizes.
    patch_segment_limit: usize,
    /// Maximum input records eligible for direct patch publication.
    patch_direct_record_limit: usize,
}

impl Default for CommitOptions {
    fn default() -> Self {
        Self {
            target_block_size: 16 * 1024,
            value_payload_compression_policy: ValuePayloadCompressionPolicy::DEFAULT,
            flush_threshold_records: 4_096,
            flush_threshold_bytes: 8 * 1024 * 1024,
            patch_segment_limit: DEFAULT_PATCH_SEGMENT_LIMIT,
            patch_direct_record_limit: DEFAULT_PATCH_DIRECT_RECORD_LIMIT,
        }
    }
}

impl CommitOptions {
    /// Sets the target logical block split size for newly written segments.
    pub fn with_target_block_size(mut self, target_block_size: usize) -> Self {
        self.target_block_size = target_block_size;
        self
    }

    /// Sets the writer-side policy for deciding whether value payloads are worth compressing.
    ///
    /// The store's persisted compression kind still controls which frame
    /// encodings are supported. This policy only affects newly written blocks
    /// whose store was created with a compression-capable kind.
    pub fn with_value_payload_compression_policy(
        mut self,
        policy: ValuePayloadCompressionPolicy,
    ) -> Self {
        self.value_payload_compression_policy = policy;
        self
    }

    /// Sets the maximum records written to one segment chunk during this commit.
    pub fn with_flush_threshold_records(mut self, flush_threshold_records: usize) -> Self {
        self.flush_threshold_records = flush_threshold_records;
        self
    }

    /// Sets the approximate maximum key/value bytes written to one segment chunk.
    pub fn with_flush_threshold_bytes(mut self, flush_threshold_bytes: usize) -> Self {
        self.flush_threshold_bytes = flush_threshold_bytes;
        self
    }

    /// Sets the maximum live patch segments before normalization is forced.
    ///
    /// A value of `0` disables direct patch publication for overlapping writes:
    /// every overlapping commit normalizes immediately.
    pub fn with_patch_segment_limit(mut self, patch_segment_limit: usize) -> Self {
        self.patch_segment_limit = patch_segment_limit;
        self
    }

    /// Sets the maximum input records eligible for direct patch publication.
    ///
    /// A value of `0` disables direct patch publication for overlapping writes:
    /// every overlapping commit normalizes immediately.
    pub fn with_patch_direct_record_limit(mut self, patch_direct_record_limit: usize) -> Self {
        self.patch_direct_record_limit = patch_direct_record_limit;
        self
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if self.target_block_size > u32::MAX as usize {
            return Err(OptionsError::TargetBlockSizeTooLarge.into());
        }
        if !self
            .value_payload_compression_policy
            .min_saved_percent_is_valid()
        {
            return Err(OptionsError::CompressionMinSavedPercentTooLarge.into());
        }
        if self.flush_threshold_records == 0 {
            return Err(OptionsError::FlushThresholdRecordsZero.into());
        }
        if self.flush_threshold_bytes == 0 {
            return Err(OptionsError::FlushThresholdBytesZero.into());
        }
        Ok(())
    }

    pub(super) fn flush_threshold_records(&self) -> usize {
        self.flush_threshold_records
    }

    pub(super) fn flush_threshold_bytes(&self) -> usize {
        self.flush_threshold_bytes
    }
}

/// Summary returned after a successful batch commit.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CommitStats {
    /// Records accepted from the caller's batch.
    pub records: usize,
    /// Sum of key and value bytes accepted from the caller.
    pub bytes: usize,
    /// Segment files made visible by the manifest update.
    pub segments_published: usize,
    /// Manifest entries removed by this publication: the rebuilt run that the
    /// batch normalized with, plus any dead entries dropped along the way.
    pub segments_retired: usize,
    /// Logical records written into newly published segments. Equals `records`
    /// for direct main and patch publishes; exceeds `records` when the commit
    /// normalized patch/main overlap into replacement main segments.
    pub merged_records: usize,
}

/// A segment written during this commit, ready to be published.
pub(super) struct WrittenSegment {
    entry: SegmentManifestEntry,
    runtime: Arc<SegmentState>,
}

/// Immutable decision made from one manifest/runtime snapshot before any file is written.
///
/// `CommitPlan` owns the manifest snapshot, the live runtime segments from that
/// snapshot, and the replacement set. The write path then has a simple phase
/// order: build plan, materialize merged entries, write files, convert the plan
/// into the next visible snapshot, publish it.
pub(super) struct CommitPlan {
    manifest: StoreManifest,
    main_segments: Vec<Arc<SegmentState>>,
    patch_segments: Vec<Arc<SegmentState>>,
    removed_ids: BTreeSet<u32>,
    next_segment_id: u32,
}

struct CommitPublication {
    manifest: StoreManifest,
    main_runtime: Vec<Arc<SegmentState>>,
    patch_runtime: Vec<Arc<SegmentState>>,
    segments_published: usize,
    segments_retired: usize,
}

pub(super) struct CommitPublicationStats {
    pub(super) segments_published: usize,
    pub(super) segments_retired: usize,
}

impl CommitPlan {
    pub(super) fn from_snapshot(
        manifest: StoreManifest,
        main_segments: Vec<Arc<SegmentState>>,
        patch_segments: Vec<Arc<SegmentState>>,
    ) -> Self {
        let live_ids: HashSet<u32> = main_segments
            .iter()
            .chain(patch_segments.iter())
            .map(|segment| segment.segment_id)
            .collect();
        let mut removed_ids = BTreeSet::new();
        for entry in &manifest.segments {
            if !live_ids.contains(&entry.segment_id) {
                removed_ids.insert(entry.segment_id);
            }
        }
        let next_segment_id = manifest.next_segment_id;
        Self {
            manifest,
            main_segments,
            patch_segments,
            removed_ids,
            next_segment_id,
        }
    }

    fn affected_main_range(&self, batch_min: &[u8], batch_max: &[u8]) -> Range<usize> {
        affected_range(self.main_entries(), batch_min, batch_max)
    }

    fn affected_main_segments(&self, range: Range<usize>) -> Vec<Arc<SegmentState>> {
        self.main_entries()[range]
            .iter()
            .filter_map(|entry| find_live_segment(&self.main_segments, entry.segment_id))
            .collect()
    }

    fn patch_segments(&self) -> Vec<Arc<SegmentState>> {
        self.patch_segments.iter().map(Arc::clone).collect()
    }

    fn should_publish_patch(
        &self,
        input_records: usize,
        new_segment_count: usize,
        options: &CommitOptions,
    ) -> bool {
        input_records <= options.patch_direct_record_limit
            && self.patch_segments.len().saturating_add(new_segment_count)
                <= options.patch_segment_limit
    }

    fn normalization_bounds(&self, batch_min: &[u8], batch_max: &[u8]) -> (Vec<u8>, Vec<u8>) {
        let mut min_key = batch_min.to_vec();
        let mut max_key = batch_max.to_vec();
        for segment in &self.patch_segments {
            if segment.min_key < min_key {
                min_key = segment.min_key.clone();
            }
            if segment.max_key > max_key {
                max_key = segment.max_key.clone();
            }
        }
        (min_key, max_key)
    }

    fn patch_bounds(&self) -> Option<(Vec<u8>, Vec<u8>)> {
        let first = self.patch_segments.first()?;
        let mut min_key = first.min_key.clone();
        let mut max_key = first.max_key.clone();
        for segment in &self.patch_segments[1..] {
            if segment.min_key < min_key {
                min_key = segment.min_key.clone();
            }
            if segment.max_key > max_key {
                max_key = segment.max_key.clone();
            }
        }
        Some((min_key, max_key))
    }

    pub(super) fn retire_normalized_segments(
        &mut self,
        region_min: &[u8],
        region_max: &[u8],
    ) -> Vec<Arc<SegmentState>> {
        let (normalize_min, normalize_max) = self.normalization_bounds(region_min, region_max);
        self.retire_live_segments_in_range(&normalize_min, &normalize_max)
    }

    fn retire_patch_normalization_segments(&mut self) -> Option<Vec<Arc<SegmentState>>> {
        let (normalize_min, normalize_max) = self.patch_bounds()?;
        Some(self.retire_live_segments_in_range(&normalize_min, &normalize_max))
    }

    fn retire_live_segments_in_range(
        &mut self,
        normalize_min: &[u8],
        normalize_max: &[u8],
    ) -> Vec<Arc<SegmentState>> {
        let affected_main = self.affected_main_range(normalize_min, normalize_max);
        let mut affected_live = self.affected_main_segments(affected_main);
        affected_live.extend(self.patch_segments());
        self.retire_segments(&affected_live);
        affected_live
    }

    fn retire_segments(&mut self, segments: &[Arc<SegmentState>]) {
        self.removed_ids
            .extend(segments.iter().map(|segment| segment.segment_id));
    }

    fn has_dead_entries(&self) -> bool {
        !self.removed_ids.is_empty()
    }

    pub(super) fn allocate_segment_id(&mut self) -> Result<u32> {
        let segment_id = self.next_segment_id;
        self.next_segment_id = self
            .next_segment_id
            .checked_add(1)
            .ok_or(CatalogMismatch::SegmentIdExhausted)?;
        Ok(segment_id)
    }

    fn into_publication(
        self,
        written: Vec<WrittenSegment>,
        key_len: usize,
    ) -> Result<CommitPublication> {
        let segments_published = written.len();
        let segments_retired = self.removed_ids.len();
        let mut manifest = self.manifest;
        manifest
            .segments
            .retain(|entry| !self.removed_ids.contains(&entry.segment_id));
        for segment in &written {
            validate_segment_entry_shape(&segment.entry, key_len)?;
            manifest.segments.push(segment.entry.clone());
        }
        sort_manifest_entries(&mut manifest.segments);
        manifest.next_segment_id = self.next_segment_id;
        manifest.validate_structure(key_len)?;

        let mut main_runtime: Vec<Arc<SegmentState>> = self
            .main_segments
            .into_iter()
            .filter(|segment| !self.removed_ids.contains(&segment.segment_id))
            .collect();
        let mut patch_runtime: Vec<Arc<SegmentState>> = self
            .patch_segments
            .into_iter()
            .filter(|segment| !self.removed_ids.contains(&segment.segment_id))
            .collect();
        for segment in &written {
            match segment.entry.tier {
                SegmentTier::Main => main_runtime.push(Arc::clone(&segment.runtime)),
                SegmentTier::Patch => patch_runtime.push(Arc::clone(&segment.runtime)),
            }
        }
        sort_runtime_segments(&mut main_runtime);
        sort_runtime_segments(&mut patch_runtime);

        Ok(CommitPublication {
            manifest,
            main_runtime,
            patch_runtime,
            segments_published,
            segments_retired,
        })
    }

    fn main_entries(&self) -> &[SegmentManifestEntry] {
        let main_count = self
            .manifest
            .segments
            .partition_point(SegmentManifestEntry::is_main);
        &self.manifest.segments[..main_count]
    }
}

impl Store {
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
        options.validate()?;
        if batch.is_empty() {
            return Ok(CommitStats::default());
        }
        let _commit_guard = self.inner.commit_lock.lock();

        let geometry = self.inner.geometry;
        let key_len = geometry.key_len;
        let mut batch = batch;
        batch.validate_lengths(key_len, geometry.value_layout)?;
        if batch.sort_and_check_duplicate_keys() {
            return Err(InputError::DuplicateKeyInBatch.into());
        }

        let input_records = batch.len();
        let input_bytes = batch.bytes;

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
            options.flush_threshold_records,
            options.flush_threshold_bytes,
        );

        let (written, merged_records) = if affected_main.is_empty() {
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
            records: input_records,
            bytes: input_bytes,
            segments_published: publication_stats.segments_published,
            segments_retired: publication_stats.segments_retired,
            merged_records,
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
        options.validate()?;
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
        if plan.patch_segments.is_empty() && !plan.has_dead_entries() {
            return Ok(CommitStats::default());
        }

        let (written, merged_records) =
            self.write_normalized_segments(&mut plan, &WriteBatch::default(), None, options)?;
        let publication_stats = self.publish_plan(plan, written, key_len)?;
        Ok(CommitStats {
            records: 0,
            bytes: 0,
            segments_published: publication_stats.segments_published,
            segments_retired: publication_stats.segments_retired,
            merged_records,
        })
    }

    fn write_normalized_segments(
        &self,
        plan: &mut CommitPlan,
        batch: &WriteBatch,
        batch_bounds: Option<(&[u8], &[u8])>,
        options: &CommitOptions,
    ) -> Result<(Vec<WrittenSegment>, usize)> {
        let affected_live = if let Some((batch_min, batch_max)) = batch_bounds {
            plan.retire_normalized_segments(batch_min, batch_max)
        } else {
            let Some(affected_live) = plan.retire_patch_normalization_segments() else {
                return Ok((Vec::new(), 0));
            };
            affected_live
        };

        let merged = self.merge_region(batch, &affected_live)?;
        let ranges = merged.flush_ranges(
            self.inner.geometry.key_len,
            options.flush_threshold_records,
            options.flush_threshold_bytes,
        );
        let written =
            self.write_batch_segments(&merged, ranges, SegmentTier::Main, plan, options)?;
        Ok((written, merged.len()))
    }

    pub(super) fn publish_plan(
        &self,
        plan: CommitPlan,
        written: Vec<WrittenSegment>,
        key_len: usize,
    ) -> Result<CommitPublicationStats> {
        let publication = plan.into_publication(written, key_len)?;
        paths::publish_manifest(&self.inner.paths, &publication.manifest)?;

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
        batch: &WriteBatch,
        affected: &[Arc<SegmentState>],
    ) -> Result<WriteBatch> {
        let geometry = self.inner.geometry;
        let verify = self.inner.verify_block_checksums;
        let mut cursors = Vec::with_capacity(affected.len());
        for segment in affected {
            cursors.push(SegmentRangeCursor::new(
                Arc::clone(segment),
                geometry,
                verify,
                None,
                None,
            )?);
        }
        let mut existing = RangeCursor::merge(cursors);

        let mut merged = WriteBatch::default();
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
                                merged.push(entry.key(), entry.value())?;
                                Advance::Batch
                            }
                            Ordering::Greater => {
                                merged.push(existing_record.key, existing_record.value)?;
                                Advance::Existing
                            }
                            Ordering::Equal => {
                                let winner = if entry.value() <= existing_record.value {
                                    entry.value()
                                } else {
                                    existing_record.value
                                };
                                merged.push(entry.key(), winner)?;
                                Advance::Both
                            }
                        }
                    }
                    (true, None) => {
                        let entry = batch.entry(index);
                        merged.push(entry.key(), entry.value())?;
                        Advance::Batch
                    }
                    (false, Some(existing_record)) => {
                        merged.push(existing_record.key, existing_record.value)?;
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
        Ok(merged.mark_sorted())
    }

    /// Writes one segment file from a contiguous range of a merged batch.
    fn write_batch_segments(
        &self,
        batch: &WriteBatch,
        ranges: Vec<Range<usize>>,
        tier: SegmentTier,
        plan: &mut CommitPlan,
        options: &CommitOptions,
    ) -> Result<Vec<WrittenSegment>> {
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
        batch: &WriteBatch,
        range: Range<usize>,
        segment_id: u32,
        tier: SegmentTier,
        options: &CommitOptions,
    ) -> Result<WrittenSegment> {
        let segment_entries = batch.view(range);
        let segment_paths = self.inner.paths.segment_publish_path(segment_id);
        let writer = self.segment_writer(options);
        let (footer, fingerprint) = segment_paths.publish().write_with(|file| {
            let mut file = SegmentFingerprintWriter::new(file);
            let footer = writer.write(&mut file, &segment_entries)?;
            Ok((footer, file.fingerprint()))
        })?;
        let min_key = footer.min_key;
        let max_key = footer.max_key;
        let block_index = footer.block_index;
        let file = fs::File::open(segment_paths.final_path())?;
        let runtime = Arc::new(SegmentState::from_written(
            segment_id,
            file,
            min_key.clone(),
            max_key.clone(),
            block_index,
        ));
        let entry = SegmentManifestEntry::new(segment_id, tier, fingerprint, min_key, max_key);
        Ok(WrittenSegment { entry, runtime })
    }

    #[cfg(any(feature = "value-compression-lz4", feature = "value-compression-zstd"))]
    fn segment_writer(&self, options: &CommitOptions) -> SegmentWriter {
        let geometry = self.inner.geometry;
        SegmentWriter::new(
            geometry.key_len,
            geometry.value_layout,
            geometry.block_checksum,
            geometry.value_payload_compression,
            options.value_payload_compression_policy,
            options.target_block_size,
        )
    }

    #[cfg(not(any(feature = "value-compression-lz4", feature = "value-compression-zstd")))]
    fn segment_writer(&self, options: &CommitOptions) -> SegmentWriter {
        let geometry = self.inner.geometry;
        SegmentWriter::new(
            geometry.key_len,
            geometry.value_layout,
            geometry.block_checksum,
            geometry.value_payload_compression,
            options.target_block_size,
        )
    }
}

/// Returns the contiguous index range of segments whose key ranges intersect
/// `[batch_min, batch_max]`. `main_entries` is sorted by `min_key` and
/// globally non-overlapping, so the intersecting set is contiguous.
fn affected_range(
    main_entries: &[SegmentManifestEntry],
    batch_min: &[u8],
    batch_max: &[u8],
) -> Range<usize> {
    let start = main_entries.partition_point(|entry| entry.max_key.as_slice() < batch_min);
    let end = main_entries.partition_point(|entry| entry.min_key.as_slice() <= batch_max);
    if start >= end {
        start..start
    } else {
        start..end
    }
}

fn sort_manifest_entries(entries: &mut [SegmentManifestEntry]) {
    entries.sort_by(|left, right| match (left.tier, right.tier) {
        (SegmentTier::Main, SegmentTier::Patch) => Ordering::Less,
        (SegmentTier::Patch, SegmentTier::Main) => Ordering::Greater,
        _ => left
            .min_key
            .cmp(&right.min_key)
            .then(left.segment_id.cmp(&right.segment_id)),
    });
}

fn sort_runtime_segments(segments: &mut [Arc<SegmentState>]) {
    segments.sort_by(|left, right| {
        left.min_key
            .cmp(&right.min_key)
            .then(left.segment_id.cmp(&right.segment_id))
    });
}

fn find_live_segment(
    live_segments: &[Arc<SegmentState>],
    segment_id: u32,
) -> Option<Arc<SegmentState>> {
    live_segments
        .iter()
        .find(|segment| segment.segment_id == segment_id)
        .map(Arc::clone)
}
