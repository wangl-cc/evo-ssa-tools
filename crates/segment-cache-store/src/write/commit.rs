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
        manifest::{SegmentManifestEntry, StoreManifest, validate_segment_entry_shape},
        record::EntrySource,
        segment::SegmentWriter,
    },
    read::cursor::{RangeCursor, SegmentRangeCursor},
    store::Store,
    write::batch::WriteBatch,
};

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
    /// batch interleaved with, plus any dead entries dropped along the way.
    pub segments_retired: usize,
    /// Records written into the replacement segments. Exceeds `records` when
    /// the commit rebuilt an intersecting region; the difference is the rewrite
    /// amplification paid for this batch's key spread.
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
    live_segments: Vec<Arc<SegmentState>>,
    affected: Range<usize>,
    removed_ids: BTreeSet<u32>,
    next_segment_id: u32,
}

struct CommitPublication {
    manifest: StoreManifest,
    runtime: Vec<Arc<SegmentState>>,
    segments_published: usize,
    segments_retired: usize,
}

impl CommitPlan {
    fn from_snapshot(
        manifest: StoreManifest,
        live_segments: Vec<Arc<SegmentState>>,
        batch_min: &[u8],
        batch_max: &[u8],
    ) -> Self {
        let affected = affected_range(&manifest, batch_min, batch_max);
        let live_ids: HashSet<u32> = live_segments
            .iter()
            .map(|segment| segment.segment_id)
            .collect();
        let mut removed_ids: BTreeSet<u32> = manifest.segments[affected.clone()]
            .iter()
            .map(|entry| entry.segment_id)
            .collect();
        for entry in &manifest.segments {
            if !live_ids.contains(&entry.segment_id) {
                removed_ids.insert(entry.segment_id);
            }
        }
        let next_segment_id = manifest.next_segment_id;
        Self {
            manifest,
            live_segments,
            affected,
            removed_ids,
            next_segment_id,
        }
    }

    fn affected_live_segments(&self) -> Vec<Arc<SegmentState>> {
        self.manifest.segments[self.affected.clone()]
            .iter()
            .filter_map(|entry| find_live_segment(&self.live_segments, entry.segment_id))
            .collect()
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
            insert_non_overlapping(&mut manifest, segment.entry.clone(), key_len)?;
        }
        manifest.next_segment_id = self.next_segment_id;

        let mut runtime: Vec<Arc<SegmentState>> = self
            .live_segments
            .into_iter()
            .filter(|segment| !self.removed_ids.contains(&segment.segment_id))
            .collect();
        for segment in &written {
            let index =
                runtime.partition_point(|existing| existing.min_key < segment.runtime.min_key);
            runtime.insert(index, Arc::clone(&segment.runtime));
        }

        Ok(CommitPublication {
            manifest,
            runtime,
            segments_published,
            segments_retired,
        })
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
        batch.sort_if_needed();
        if batch.has_duplicate_keys() {
            return Err(InputError::DuplicateKeyInBatch.into());
        }

        let input_records = batch.len();
        let input_bytes = batch.bytes;

        let (manifest, live_segments) = {
            let state = self.inner.state.read();
            (state.manifest.clone(), state.segments_as_vec())
        };

        let batch_min = batch.key_at(0).to_vec();
        let batch_max = batch.key_at(input_records - 1).to_vec();
        let mut plan = CommitPlan::from_snapshot(manifest, live_segments, &batch_min, &batch_max);

        let merged = if plan.affected.is_empty() {
            batch
        } else {
            let affected_live = plan.affected_live_segments();
            self.merge_region(&batch, &affected_live)?
        };

        // Write the merged region into fresh segments.
        let ranges = merged.flush_ranges(
            key_len,
            options.flush_threshold_records,
            options.flush_threshold_bytes,
        );
        let mut written = Vec::with_capacity(ranges.len());
        for range in ranges {
            let segment_id = plan.allocate_segment_id()?;
            written.push(self.write_segment(&merged, range, segment_id, options)?);
        }

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
            state.replace_segments(publication.runtime);
            state.manifest = publication.manifest;
        }

        Ok(CommitStats {
            records: input_records,
            bytes: input_bytes,
            segments_published,
            segments_retired,
            merged_records: merged.len(),
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
        let mut existing = RangeCursor::new(cursors);

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
    fn write_segment(
        &self,
        batch: &WriteBatch,
        range: Range<usize>,
        segment_id: u32,
        options: &CommitOptions,
    ) -> Result<WrittenSegment> {
        let geometry = self.inner.geometry;
        let segment_entries = batch.view(range);
        let segment_paths = self.inner.paths.segment_publish_path(segment_id);
        let (footer, block_index) = segment_paths.publish().write_with(|file| {
            Ok(SegmentWriter::new(
                geometry.key_len,
                geometry.value_layout,
                options.target_block_size,
            )
            .write(file, &segment_entries)?)
        })?;
        let file = fs::File::open(segment_paths.final_path())?;
        let runtime = Arc::new(SegmentState::from_written(
            segment_id,
            file,
            footer.min_key.clone(),
            footer.max_key.clone(),
            block_index,
        ));
        let entry = SegmentManifestEntry {
            segment_id,
            min_key: footer.min_key,
            max_key: footer.max_key,
        };
        Ok(WrittenSegment { entry, runtime })
    }
}

/// Returns the contiguous index range of segments whose key ranges intersect
/// `[batch_min, batch_max]`. The slice `manifest.segments` is sorted by `min_key`
/// and globally non-overlapping, so the intersecting set is contiguous.
fn affected_range(manifest: &StoreManifest, batch_min: &[u8], batch_max: &[u8]) -> Range<usize> {
    let segments = &manifest.segments;
    let start = segments.partition_point(|entry| entry.max_key.as_slice() < batch_min);
    let end = segments.partition_point(|entry| entry.min_key.as_slice() <= batch_max);
    if start >= end {
        start..start
    } else {
        start..end
    }
}

/// Inserts a new entry into the manifest's sorted segment list, rejecting any
/// overlap with a neighbor.
///
/// This is commit policy, not manifest format: the format layer validates a
/// parsed manifest as a whole, while this guards one insertion during a
/// replacing commit.
fn insert_non_overlapping(
    manifest: &mut StoreManifest,
    entry: SegmentManifestEntry,
    key_len: usize,
) -> Result<usize> {
    let index = manifest
        .segments
        .partition_point(|existing| existing.min_key.as_slice() < entry.min_key.as_slice());
    validate_segment_entry_shape(&entry, key_len)?;
    if let Some(previous) = index
        .checked_sub(1)
        .map(|previous| &manifest.segments[previous])
        && entry.min_key.as_slice() <= previous.max_key.as_slice()
    {
        return Err(InputError::SegmentOverlap.into());
    }
    if let Some(next) = manifest.segments.get(index)
        && entry.max_key.as_slice() >= next.min_key.as_slice()
    {
        return Err(InputError::SegmentOverlap.into());
    }
    manifest.segments.insert(index, entry);
    Ok(index)
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
