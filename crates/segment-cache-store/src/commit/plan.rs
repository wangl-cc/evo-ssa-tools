//! Pure planning for one atomic manifest transition.

use std::{
    cmp::Ordering,
    collections::{BTreeSet, HashSet},
    ops::Range,
    sync::Arc,
};

use super::options::CommitOptions;
use crate::{
    catalog::{CatalogMismatch, SegmentManifestEntry, SegmentTier, StoreManifest},
    error::Result,
    segment::Segment,
};

/// Segment file materialized by a commit but not yet visible in the manifest.
pub(super) struct StagedSegment {
    entry: SegmentManifestEntry,
    runtime: Arc<Segment>,
}

impl StagedSegment {
    pub(super) fn new(entry: SegmentManifestEntry, runtime: Arc<Segment>) -> Self {
        Self { entry, runtime }
    }
}

/// Immutable decision made from one manifest/runtime snapshot before any file is written.
///
/// The plan owns the source snapshot and replacement set. Publication consumes
/// it exactly once to construct the next manifest and visible segment lists.
pub(super) struct CommitPlan {
    manifest: StoreManifest,
    main_segments: Vec<Arc<Segment>>,
    pub(super) patch_segments: Vec<Arc<Segment>>,
    removed_ids: BTreeSet<u32>,
    next_segment_id: u32,
}

pub(super) struct CommitPublication {
    pub(super) manifest: StoreManifest,
    pub(super) main_runtime: Vec<Arc<Segment>>,
    pub(super) patch_runtime: Vec<Arc<Segment>>,
    pub(super) segments_published: usize,
    pub(super) segments_retired: usize,
}

pub(super) struct CommitPublicationStats {
    pub(super) segments_published: usize,
    pub(super) segments_retired: usize,
}

pub(super) struct RetiredSegments {
    pub(super) main: Vec<Arc<Segment>>,
    pub(super) patches: Vec<Arc<Segment>>,
}

impl CommitPlan {
    pub(super) fn from_snapshot(
        manifest: StoreManifest,
        main_segments: Vec<Arc<Segment>>,
        patch_segments: Vec<Arc<Segment>>,
    ) -> Self {
        let live_ids: HashSet<u32> = main_segments
            .iter()
            .chain(patch_segments.iter())
            .map(|segment| segment.id())
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

    pub(super) fn affected_main_range(&self, batch_min: &[u8], batch_max: &[u8]) -> Range<usize> {
        affected_range(self.main_entries(), batch_min, batch_max)
    }

    fn affected_main_segments(&self, range: Range<usize>) -> Vec<Arc<Segment>> {
        self.main_entries()[range]
            .iter()
            .filter_map(|entry| find_live_segment(&self.main_segments, entry.segment_id))
            .collect()
    }

    fn patch_segments(&self) -> Vec<Arc<Segment>> {
        self.patch_segments.iter().map(Arc::clone).collect()
    }

    pub(super) fn has_patches(&self) -> bool {
        !self.patch_segments.is_empty()
    }

    pub(super) fn should_publish_patch(
        &self,
        input_records: usize,
        new_segment_count: usize,
        options: &CommitOptions,
    ) -> bool {
        input_records <= options.patch_direct_record_limit()
            && self.patch_segments.len().saturating_add(new_segment_count)
                <= options.patch_segment_limit()
    }

    fn normalization_bounds(&self, batch_min: &[u8], batch_max: &[u8]) -> (Vec<u8>, Vec<u8>) {
        let mut min_key = batch_min.to_vec();
        let mut max_key = batch_max.to_vec();
        for segment in &self.patch_segments {
            if segment.min_key() < min_key.as_slice() {
                min_key = segment.min_key().to_vec();
            }
            if segment.max_key() > max_key.as_slice() {
                max_key = segment.max_key().to_vec();
            }
        }
        (min_key, max_key)
    }

    fn patch_bounds(&self) -> Option<(Vec<u8>, Vec<u8>)> {
        let first = self.patch_segments.first()?;
        let mut min_key = first.min_key().to_vec();
        let mut max_key = first.max_key().to_vec();
        for segment in &self.patch_segments[1..] {
            if segment.min_key() < min_key.as_slice() {
                min_key = segment.min_key().to_vec();
            }
            if segment.max_key() > max_key.as_slice() {
                max_key = segment.max_key().to_vec();
            }
        }
        Some((min_key, max_key))
    }

    pub(super) fn retire_normalized_segments(
        &mut self,
        region_min: &[u8],
        region_max: &[u8],
    ) -> RetiredSegments {
        let (normalize_min, normalize_max) = self.normalization_bounds(region_min, region_max);
        self.retire_live_segments_in_range(&normalize_min, &normalize_max)
    }

    pub(super) fn retire_patch_normalization_segments(&mut self) -> Option<RetiredSegments> {
        let (normalize_min, normalize_max) = self.patch_bounds()?;
        Some(self.retire_live_segments_in_range(&normalize_min, &normalize_max))
    }

    fn retire_live_segments_in_range(
        &mut self,
        normalize_min: &[u8],
        normalize_max: &[u8],
    ) -> RetiredSegments {
        let affected_main = self.affected_main_range(normalize_min, normalize_max);
        let main = self.affected_main_segments(affected_main);
        let patches = self.patch_segments();
        self.retire_segments(&main);
        self.retire_segments(&patches);
        RetiredSegments { main, patches }
    }

    fn retire_segments(&mut self, segments: &[Arc<Segment>]) {
        self.removed_ids
            .extend(segments.iter().map(|segment| segment.id()));
    }

    pub(super) fn has_dead_entries(&self) -> bool {
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

    pub(super) fn into_publication(
        self,
        staged: Vec<StagedSegment>,
        key_len: usize,
    ) -> Result<CommitPublication> {
        let segments_published = staged.len();
        let segments_retired = self.removed_ids.len();
        let mut manifest = self.manifest;
        manifest
            .segments
            .retain(|entry| !self.removed_ids.contains(&entry.segment_id));
        for segment in &staged {
            segment.entry.validate_shape(key_len)?;
            manifest.segments.push(segment.entry.clone());
        }
        sort_manifest_entries(&mut manifest.segments);
        manifest.next_segment_id = self.next_segment_id;
        manifest.validate_structure(key_len)?;

        let mut main_runtime: Vec<Arc<Segment>> = self
            .main_segments
            .into_iter()
            .filter(|segment| !self.removed_ids.contains(&segment.id()))
            .collect();
        let mut patch_runtime: Vec<Arc<Segment>> = self
            .patch_segments
            .into_iter()
            .filter(|segment| !self.removed_ids.contains(&segment.id()))
            .collect();
        for segment in &staged {
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

/// Returns the contiguous main-tier range intersecting `[batch_min, batch_max]`.
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

fn sort_runtime_segments(segments: &mut [Arc<Segment>]) {
    segments.sort_by(|left, right| {
        left.min_key()
            .cmp(right.min_key())
            .then(left.id().cmp(&right.id()))
    });
}

fn find_live_segment(live_segments: &[Arc<Segment>], segment_id: u32) -> Option<Arc<Segment>> {
    live_segments
        .iter()
        .find(|segment| segment.id() == segment_id)
        .map(Arc::clone)
}
