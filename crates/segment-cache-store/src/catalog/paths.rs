//! Path layout of one store root.
//!
//! Also owns catalog file IO: loading and atomically publishing `STORE` and
//! `MANIFEST`. Byte layouts and structural validation live in
//! the catalog codecs; these functions only move those bytes to and from disk.

use std::{
    fs::{self, File},
    path::{Path, PathBuf},
};

use super::{
    StoreManifest,
    descriptor::StoreDescriptor,
    io::{AtomicFilePublish, StagedFileBatch, temp_path_for},
};
use crate::Result;

const STORE_FILE_NAME: &str = "STORE";
const MANIFEST_FILE_NAME: &str = "MANIFEST";
const LOCK_FILE_NAME: &str = "LOCK";

/// Owned path bundle for all stable files and directories in one store root.
pub(crate) struct StorePaths {
    root: PathBuf,
    store_file: PathBuf,
    manifest: PathBuf,
    segment_dir: PathBuf,
    lock_file: PathBuf,
}

impl StorePaths {
    pub(crate) fn new(root: impl AsRef<Path>) -> Self {
        let root = root.as_ref().to_path_buf();
        let segment_dir = root.join("segments");
        Self {
            store_file: root.join(STORE_FILE_NAME),
            manifest: root.join(MANIFEST_FILE_NAME),
            segment_dir,
            lock_file: root.join(LOCK_FILE_NAME),
            root,
        }
    }

    pub(crate) fn lock_file(&self) -> &Path {
        &self.lock_file
    }

    pub(crate) fn root(&self) -> &Path {
        &self.root
    }

    pub(crate) fn store_file(&self) -> &Path {
        &self.store_file
    }

    pub(crate) fn segment_dir(&self) -> &Path {
        &self.segment_dir
    }

    pub(crate) fn ensure_root(&self) -> Result<()> {
        fs::create_dir_all(&self.root)?;
        Ok(())
    }

    pub(crate) fn ensure_dirs(&self) -> Result<()> {
        self.ensure_root()?;
        fs::create_dir_all(&self.segment_dir)?;
        Ok(())
    }

    pub(crate) fn final_segment(&self, segment_id: u32) -> PathBuf {
        self.segment_dir.join(segment_file_name(segment_id))
    }

    /// Deletes leftover catalog temp files (`STORE.tmp`, `MANIFEST.tmp`).
    ///
    /// These only exist after a crash mid-publication; the next publication of
    /// the same file would overwrite them anyway. Deletion is best-effort.
    pub(crate) fn remove_stale_catalog_temps(&self) {
        let _ = fs::remove_file(temp_path_for(&self.store_file));
        let _ = fs::remove_file(temp_path_for(&self.manifest));
    }

    pub(super) fn load_descriptor(&self) -> Result<Option<StoreDescriptor>> {
        if !self.store_file.exists() {
            return Ok(None);
        }
        Ok(Some(StoreDescriptor::parse(&fs::read_to_string(
            &self.store_file,
        )?)?))
    }

    pub(super) fn publish_descriptor(&self, descriptor: &StoreDescriptor) -> Result<()> {
        publish_for(&self.store_file).write_bytes(descriptor.encode().as_bytes())
    }

    pub(super) fn load_manifest(&self) -> Result<Option<StoreManifest>> {
        if !self.manifest.exists() {
            return Ok(None);
        }
        Ok(Some(StoreManifest::parse(&fs::read(&self.manifest)?)?))
    }

    pub(crate) fn publish_manifest(&self, manifest: &StoreManifest) -> Result<()> {
        publish_for(&self.manifest).write_bytes(&manifest.encode()?)
    }

    pub(crate) fn segment_batch(&self) -> SegmentBatchPublication {
        SegmentBatchPublication {
            segment_dir: self.segment_dir.clone(),
            files: StagedFileBatch::new(&self.segment_dir),
        }
    }
}

/// File name for one segment id; file names are opaque identities.
pub(super) fn segment_file_name(segment_id: u32) -> String {
    // 10 digits cover the full u32 id range.
    format!("segment-{segment_id:010}.seg")
}

/// Segment files staged under one shared durability barrier.
pub(crate) struct SegmentBatchPublication {
    segment_dir: PathBuf,
    files: StagedFileBatch,
}

impl SegmentBatchPublication {
    pub(crate) fn stage_with<T>(
        &mut self,
        segment_id: u32,
        write: impl FnOnce(&mut File) -> Result<T>,
    ) -> Result<T> {
        let final_path = self.segment_dir.join(segment_file_name(segment_id));
        self.files.stage_with(final_path, write)
    }

    pub(crate) fn publish(self) -> Result<()> {
        self.files.publish()
    }
}

fn publish_for(path: &Path) -> AtomicFilePublish<'_> {
    let Some(publish) = AtomicFilePublish::new(path) else {
        unreachable!("store paths are named files under the store root");
    };
    publish
}
