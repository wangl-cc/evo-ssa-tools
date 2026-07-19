//! Shared state behind the cheaply cloneable public [`crate::Store`] handle.

use parking_lot::{Mutex, RwLock};

use crate::{
    catalog::{OpenedStore, StoreMetadata, StorePaths, VisibleSnapshot, WriterLock},
    segment::SegmentGeometry,
};

pub(crate) struct StoreInner {
    pub(crate) paths: StorePaths,
    pub(crate) metadata: StoreMetadata,
    pub(crate) geometry: SegmentGeometry,
    pub(crate) verify_block_checksums: bool,
    pub(crate) commit_lock: Mutex<()>,
    pub(crate) state: RwLock<VisibleSnapshot>,
    /// Held for the store's lifetime by a writer open; `None` for read-only opens.
    pub(crate) writer_lock: Option<WriterLock>,
}

impl StoreInner {
    pub(crate) fn from_opened(opened: OpenedStore) -> Self {
        Self {
            paths: opened.paths,
            metadata: opened.metadata,
            geometry: opened.geometry,
            verify_block_checksums: opened.verify_block_checksums,
            commit_lock: Mutex::new(()),
            state: RwLock::new(opened.snapshot),
            writer_lock: opened.writer_lock,
        }
    }
}
