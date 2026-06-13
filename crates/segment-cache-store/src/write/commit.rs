//! Batch commit options and replacing-manifest segment publication.

use std::{
    cmp::Ordering,
    collections::{BTreeSet, HashSet},
    fs,
    ops::Range,
    sync::Arc,
};

use crate::{
    engine::{gc::garbage_collect_unreferenced, paths, runtime::SegmentState},
    error::{InputError, OptionsError, Result},
    format::{
        CatalogMismatch,
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

const PATCH_SEGMENT_LIMIT: usize = 8;
const PATCH_DIRECT_RECORD_LIMIT: usize = 4_096;

/// Options consumed by one batch commit.
///
/// These fields affect newly written segment files only. They are not part of
/// the namespace identity and are not required when opening an existing store.
#[derive(Clone, Debug)]
pub struct CommitOptions {
    /// Target physical block size for newly written segments.
    pub target_block_size: usize,
    /// Maximum records per newly published segment chunk.
    pub flush_threshold_records: usize,
    /// Maximum approximate key/value bytes per newly published segment chunk.
    pub flush_threshold_bytes: usize,
}

impl Default for CommitOptions {
    fn default() -> Self {
        Self {
            target_block_size: 16 * 1024,
            flush_threshold_records: 4_096,
            flush_threshold_bytes: 8 * 1024 * 1024,
        }
    }
}

impl CommitOptions {
    /// Sets the target physical block size for newly written segments.
    pub fn with_target_block_size(mut self, target_block_size: usize) -> Self {
        self.target_block_size = target_block_size;
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

    pub(crate) fn validate(&self) -> Result<()> {
        if self.target_block_size > u32::MAX as usize {
            return Err(OptionsError::TargetBlockSizeTooLarge.into());
        }
        if self.flush_threshold_records == 0 {
            return Err(OptionsError::FlushThresholdRecordsZero.into());
        }
        if self.flush_threshold_bytes == 0 {
            return Err(OptionsError::FlushThresholdBytesZero.into());
        }
        Ok(())
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
struct WrittenSegment {
    entry: SegmentManifestEntry,
    runtime: Arc<SegmentState>,
}

/// Immutable decision made from one manifest/runtime snapshot before any file is written.
///
/// `CommitPlan` owns the manifest snapshot, the live runtime segments from that
/// snapshot, and the replacement set. The write path then has a simple phase
/// order: build plan, materialize merged entries, write files, convert the plan
/// into the next visible snapshot, publish it.
struct CommitPlan {
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

impl CommitPlan {
    fn from_snapshot(
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

    fn should_publish_patch(&self, input_records: usize, new_segment_count: usize) -> bool {
        input_records <= PATCH_DIRECT_RECORD_LIMIT
            && self.patch_segments.len() + new_segment_count <= PATCH_SEGMENT_LIMIT
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

    fn retire_segments(&mut self, segments: &[Arc<SegmentState>]) {
        self.removed_ids
            .extend(segments.iter().map(|segment| segment.segment_id));
    }

    fn allocate_segment_id(&mut self) -> Result<u32> {
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
        } else if plan.should_publish_patch(input_records, direct_ranges.len()) {
            let written = self.write_batch_segments(
                &batch,
                direct_ranges,
                SegmentTier::Patch,
                &mut plan,
                options,
            )?;
            (written, input_records)
        } else {
            let (normalize_min, normalize_max) = plan.normalization_bounds(&batch_min, &batch_max);
            let affected_main = plan.affected_main_range(&normalize_min, &normalize_max);
            let mut affected_live = plan.affected_main_segments(affected_main);
            affected_live.extend(plan.patch_segments());
            plan.retire_segments(&affected_live);
            let merged = self.merge_region(&batch, &affected_live)?;
            let ranges = merged.flush_ranges(
                key_len,
                options.flush_threshold_records,
                options.flush_threshold_bytes,
            );
            let written =
                self.write_batch_segments(&merged, ranges, SegmentTier::Main, &mut plan, options)?;
            (written, merged.len())
        };

        let publication = plan.into_publication(written, key_len)?;
        paths::publish_manifest(&self.inner.paths, &publication.manifest)?;

        // Reclaim everything the new manifest no longer references: the run
        // this publication retired, dead entries' files, and any orphans left
        // by earlier failed publications. Best-effort: open readers may still
        // hold descriptors, and a failed unlink is retried by the next pass.
        garbage_collect_unreferenced(&self.inner.paths, &publication.manifest);
        let segments_published = publication.segments_published;
        let segments_retired = publication.segments_retired;

        {
            let mut state = self.inner.state.write();
            state.replace_segments(publication.main_runtime, publication.patch_runtime);
            state.manifest = publication.manifest;
        }

        Ok(CommitStats {
            records: input_records,
            bytes: input_bytes,
            segments_published,
            segments_retired,
            merged_records,
        })
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
                geometry.key_len,
                geometry.value_layout,
                verify,
                None,
                None,
            )?);
        }
        let mut existing = RangeCursor::merge(cursors);

        let mut merged = WriteBatch::default();
        let mut pending = next_existing(&mut existing)?;
        let mut index = 0usize;
        let len = batch.len();
        loop {
            match (index < len, pending.as_ref()) {
                (true, Some((existing_key, _))) => {
                    let entry = batch.entry(index);
                    match entry.key().cmp(existing_key.as_slice()) {
                        Ordering::Less => {
                            merged.push(entry.key(), entry.value())?;
                            index += 1;
                        }
                        Ordering::Greater => {
                            let (key, value) = pending.take().expect("pending was Some");
                            merged.push_owned(key, value)?;
                            pending = next_existing(&mut existing)?;
                        }
                        Ordering::Equal => {
                            let (key, existing_value) = pending.take().expect("pending was Some");
                            let winner = if entry.value() <= existing_value.as_slice() {
                                entry.value().to_vec()
                            } else {
                                existing_value
                            };
                            merged.push_owned(key, winner)?;
                            index += 1;
                            pending = next_existing(&mut existing)?;
                        }
                    }
                }
                (true, None) => {
                    let entry = batch.entry(index);
                    merged.push(entry.key(), entry.value())?;
                    index += 1;
                }
                (false, Some(_)) => {
                    let (key, value) = pending.take().expect("pending was Some");
                    merged.push_owned(key, value)?;
                    pending = next_existing(&mut existing)?;
                }
                (false, None) => break,
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
    fn write_segment(
        &self,
        batch: &WriteBatch,
        range: Range<usize>,
        segment_id: u32,
        tier: SegmentTier,
        options: &CommitOptions,
    ) -> Result<WrittenSegment> {
        let geometry = self.inner.geometry;
        let segment_entries = batch.view(range);
        let segment_paths = self.inner.paths.segment_publish_path(segment_id);
        let footer = segment_paths.publish().write_with(|file| {
            Ok(SegmentWriter::new(
                geometry.key_len,
                geometry.value_layout,
                options.target_block_size,
            )
            .write(file, &segment_entries)?)
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
        let entry = SegmentManifestEntry::new(segment_id, tier, min_key, max_key);
        Ok(WrittenSegment { entry, runtime })
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

fn next_existing(cursor: &mut RangeCursor) -> Result<Option<(Vec<u8>, Vec<u8>)>> {
    cursor.next().transpose()
}
