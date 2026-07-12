//! Store-root storage statistics.

use std::{fs, path::Path};

use crate::{Result, catalog::paths::StorePaths};

/// Physical file usage for one store root.
///
/// Segment counts and bytes include only immutable segment files owned by the
/// current store layout. Total counts and bytes include every regular file
/// under the root, including catalog files, lock files, temporary files,
/// retired segments, and orphan files.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub struct StoreStorageStats {
    /// Segment files under the store's segment-file area.
    pub segment_files: usize,
    /// Bytes occupied by segment files.
    pub segment_bytes: u64,
    /// Regular files under the store root.
    pub total_files: usize,
    /// Bytes occupied by regular files under the store root.
    pub total_bytes: u64,
}

pub(crate) fn collect_storage_stats(paths: &StorePaths) -> Result<StoreStorageStats> {
    let mut stats = StoreStorageStats::default();
    StorageStatsCollector::new(paths).collect_into(&mut stats, paths.root())?;
    Ok(stats)
}

struct StorageStatsCollector<'a> {
    paths: &'a StorePaths,
}

impl<'a> StorageStatsCollector<'a> {
    fn new(paths: &'a StorePaths) -> Self {
        Self { paths }
    }

    fn collect_into(&self, stats: &mut StoreStorageStats, path: &Path) -> std::io::Result<()> {
        let metadata = fs::symlink_metadata(path)?;
        if metadata.is_file() {
            stats.total_files += 1;
            stats.total_bytes += metadata.len();
            if self.is_segment_file(path) {
                stats.segment_files += 1;
                stats.segment_bytes += metadata.len();
            }
            return Ok(());
        }
        if metadata.is_dir() {
            for entry in fs::read_dir(path)? {
                self.collect_into(stats, &entry?.path())?;
            }
        }
        Ok(())
    }

    fn is_segment_file(&self, path: &Path) -> bool {
        path.parent() == Some(self.paths.segment_dir())
            && path.extension().is_some_and(|extension| extension == "seg")
    }
}
