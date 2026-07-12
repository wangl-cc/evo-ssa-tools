//! Shared state behind the cheaply cloneable public [`crate::Store`] handle.

use std::sync::Arc;

use parking_lot::{Mutex, RwLock};

use crate::{
    catalog::{io::WriterLock, manifest::StoreManifest, paths::StorePaths},
    schema::StoreMetadata,
};
pub(crate) use crate::{schema::StoreGeometry, segment::state::SegmentState};

pub(crate) struct StoreInner {
    pub(crate) paths: StorePaths,
    pub(crate) metadata: StoreMetadata,
    pub(crate) geometry: StoreGeometry,
    pub(crate) verify_block_checksums: bool,
    pub(crate) commit_lock: Mutex<()>,
    pub(crate) state: RwLock<StoreState>,
    /// Held for the store's lifetime by a writer open; `None` for read-only opens.
    pub(crate) writer_lock: Option<WriterLock>,
}

pub(crate) struct StoreState {
    pub(crate) manifest: StoreManifest,
    pub(crate) main_segments: SegmentSnapshot,
    pub(crate) patch_segments: SegmentSnapshot,
}

pub(crate) type SegmentSnapshot = Arc<[Arc<SegmentState>]>;

impl StoreState {
    pub(crate) fn new(
        manifest: StoreManifest,
        main_segments: Vec<Arc<SegmentState>>,
        patch_segments: Vec<Arc<SegmentState>>,
    ) -> Self {
        Self {
            manifest,
            main_segments: main_segments.into(),
            patch_segments: patch_segments.into(),
        }
    }

    pub(crate) fn main_segments_as_vec(&self) -> Vec<Arc<SegmentState>> {
        self.main_segments.iter().cloned().collect()
    }

    pub(crate) fn patch_segments_as_vec(&self) -> Vec<Arc<SegmentState>> {
        self.patch_segments.iter().cloned().collect()
    }

    pub(crate) fn replace_segments(
        &mut self,
        main_segments: Vec<Arc<SegmentState>>,
        patch_segments: Vec<Arc<SegmentState>>,
    ) {
        self.main_segments = main_segments.into();
        self.patch_segments = patch_segments.into();
    }
}
