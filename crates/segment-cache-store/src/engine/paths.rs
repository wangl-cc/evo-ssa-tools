//! Path layout of one store root.

use std::{
    fs,
    path::{Path, PathBuf},
};

use crate::{
    Result,
    engine::io::{AtomicFilePublish, temp_path_for},
};

const STORE_FILE_NAME: &str = "STORE";
const MANIFEST_FILE_NAME: &str = "MANIFEST";

/// Owned path bundle for all stable files and directories in one store root.
pub(crate) struct StorePaths {
    store_file: PathBuf,
    manifest: PathBuf,
    segment_dir: PathBuf,
}

impl StorePaths {
    pub(crate) fn new(root: impl AsRef<Path>) -> Self {
        let root = root.as_ref();
        let segment_dir = root.join("segments");
        Self {
            store_file: root.join(STORE_FILE_NAME),
            manifest: root.join(MANIFEST_FILE_NAME),
            segment_dir,
        }
    }

    pub(crate) fn store_file(&self) -> &Path {
        &self.store_file
    }

    pub(crate) fn manifest(&self) -> &Path {
        &self.manifest
    }

    pub(crate) fn segment_dir(&self) -> &Path {
        &self.segment_dir
    }

    pub(crate) fn ensure_dirs(&self) -> Result<()> {
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

    pub(crate) fn store_file_publish(&self) -> AtomicFilePublish<'_> {
        publish_for(&self.store_file)
    }

    pub(crate) fn manifest_publish(&self) -> AtomicFilePublish<'_> {
        publish_for(&self.manifest)
    }

    pub(crate) fn segment_publish_path(&self, segment_id: u32) -> SegmentPublishPath {
        SegmentPublishPath {
            final_path: self.final_segment(segment_id),
        }
    }
}

/// File name for one segment id; file names are opaque identities.
pub(crate) fn segment_file_name(segment_id: u32) -> String {
    // 10 digits cover the full u32 id range.
    format!("segment-{segment_id:010}.seg")
}

pub(crate) struct SegmentPublishPath {
    final_path: PathBuf,
}

impl SegmentPublishPath {
    pub(crate) fn final_path(&self) -> &Path {
        &self.final_path
    }

    pub(crate) fn publish(&self) -> AtomicFilePublish<'_> {
        publish_for(&self.final_path)
    }
}

fn publish_for(path: &Path) -> AtomicFilePublish<'_> {
    let Some(publish) = AtomicFilePublish::new(path) else {
        unreachable!("store paths are named files under the store root");
    };
    publish
}
