//! Pure planning for one atomic manifest transition.

use std::{
    cmp::Ordering,
    collections::{BTreeSet, HashSet},
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

pub(super) struct NormalizationComponent {
    main: Vec<Arc<Segment>>,
    patches: Vec<Arc<Segment>>,
}

impl NormalizationComponent {
    pub(super) fn has_main(&self) -> bool {
        !self.main.is_empty()
    }

    pub(super) fn has_patches(&self) -> bool {
        !self.patches.is_empty()
    }

    pub(super) fn allows_patch(
        &self,
        input_records: usize,
        input_bytes: usize,
        new_segment_count: usize,
        options: &CommitOptions,
    ) -> bool {
        self.has_main()
            && !self.has_patches()
            && new_segment_count == 1
            && input_records <= options.patch_direct_record_limit()
            && input_bytes <= options.flush_threshold_bytes()
    }
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

    pub(super) fn has_patches(&self) -> bool {
        !self.patch_segments.is_empty()
    }

    pub(super) fn component_for_range(
        &self,
        region_min: &[u8],
        region_max: &[u8],
    ) -> NormalizationComponent {
        let mut min_key = region_min.to_vec();
        let mut max_key = region_max.to_vec();
        let mut main_ids = HashSet::new();
        let mut patch_ids = HashSet::new();

        loop {
            let mut changed = false;
            for segment in &self.main_segments {
                if !main_ids.contains(&segment.id())
                    && ranges_intersect(segment.min_key(), segment.max_key(), &min_key, &max_key)
                {
                    main_ids.insert(segment.id());
                    expand_bounds(&mut min_key, &mut max_key, segment);
                    changed = true;
                }
            }
            for segment in &self.patch_segments {
                if !patch_ids.contains(&segment.id())
                    && ranges_intersect(segment.min_key(), segment.max_key(), &min_key, &max_key)
                {
                    patch_ids.insert(segment.id());
                    expand_bounds(&mut min_key, &mut max_key, segment);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        NormalizationComponent {
            main: self
                .main_segments
                .iter()
                .filter(|segment| main_ids.contains(&segment.id()))
                .cloned()
                .collect(),
            patches: self
                .patch_segments
                .iter()
                .filter(|segment| patch_ids.contains(&segment.id()))
                .cloned()
                .collect(),
        }
    }

    pub(super) fn patch_components(&self) -> Vec<NormalizationComponent> {
        let mut included_patch_ids = HashSet::new();
        let mut components = Vec::new();
        for patch in &self.patch_segments {
            if included_patch_ids.contains(&patch.id()) {
                continue;
            }
            let component = self.component_for_range(patch.min_key(), patch.max_key());
            included_patch_ids.extend(component.patches.iter().map(|segment| segment.id()));
            components.push(component);
        }
        components
    }

    pub(super) fn retire_component(
        &mut self,
        component: NormalizationComponent,
    ) -> RetiredSegments {
        self.retire_segments(&component.main);
        self.retire_segments(&component.patches);
        RetiredSegments {
            main: component.main,
            patches: component.patches,
        }
    }

    pub(super) fn retire_all_live_segments(&mut self) -> RetiredSegments {
        let main = self.main_segments.clone();
        let patches = self.patch_segments.clone();
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

fn ranges_intersect(left_min: &[u8], left_max: &[u8], right_min: &[u8], right_max: &[u8]) -> bool {
    left_min <= right_max && right_min <= left_max
}

fn expand_bounds(min_key: &mut Vec<u8>, max_key: &mut Vec<u8>, segment: &Segment) {
    if segment.min_key() < min_key.as_slice() {
        min_key.clear();
        min_key.extend_from_slice(segment.min_key());
    }
    if segment.max_key() > max_key.as_slice() {
        max_key.clear();
        max_key.extend_from_slice(segment.max_key());
    }
}
