//! One immutable manifest-selected view of visible segments.

use std::sync::Arc;

use super::StoreManifest;
use crate::segment::Segment;

pub(crate) type SegmentSnapshot = Arc<[Arc<Segment>]>;

pub(crate) struct VisibleSnapshot {
    pub(crate) manifest: StoreManifest,
    pub(crate) main_segments: SegmentSnapshot,
    pub(crate) patch_segments: SegmentSnapshot,
}

impl VisibleSnapshot {
    pub(crate) fn new(
        manifest: StoreManifest,
        main_segments: Vec<Arc<Segment>>,
        patch_segments: Vec<Arc<Segment>>,
    ) -> Self {
        Self {
            manifest,
            main_segments: main_segments.into(),
            patch_segments: patch_segments.into(),
        }
    }

    pub(crate) fn main_segments_as_vec(&self) -> Vec<Arc<Segment>> {
        self.main_segments.iter().cloned().collect()
    }

    pub(crate) fn patch_segments_as_vec(&self) -> Vec<Arc<Segment>> {
        self.patch_segments.iter().cloned().collect()
    }

    pub(crate) fn replace_segments(
        &mut self,
        main_segments: Vec<Arc<Segment>>,
        patch_segments: Vec<Arc<Segment>>,
    ) {
        self.main_segments = main_segments.into();
        self.patch_segments = patch_segments.into();
    }
}
