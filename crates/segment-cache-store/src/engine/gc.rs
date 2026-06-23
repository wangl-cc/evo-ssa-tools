//! Explicit segment garbage collection.

use std::{collections::HashSet, fs};

use crate::{
    engine::paths::{StorePaths, segment_file_name},
    format::manifest::StoreManifest,
};

/// Deletes every file under `segments/` that the manifest does not reference.
///
/// Unreferenced files are provably dead: `MANIFEST` is the sole source of
/// visibility and commits never adopt pre-existing files. This reclaims orphan
/// final segments left by a crash between rename and manifest publication, plus
/// any `.tmp` leftovers. Deletion is best-effort; failures are left for a later
/// pass and never affect visibility.
///
/// Callers must run this only from an explicit writer maintenance operation.
/// It does not run during open or commit because read-only opens intentionally
/// take no writer lock; automatic deletion could remove segment files from an
/// older manifest snapshot that a concurrent read-only open has already read
/// but not fully opened yet.
pub(crate) fn garbage_collect_unreferenced(paths: &StorePaths, manifest: &StoreManifest) {
    let referenced: HashSet<String> = manifest
        .segments
        .iter()
        .map(|entry| segment_file_name(entry.segment_id))
        .collect();

    let Ok(entries) = fs::read_dir(paths.segment_dir()) else {
        return;
    };
    for entry in entries.flatten() {
        let keep = entry
            .file_name()
            .to_str()
            .is_some_and(|name| referenced.contains(name));
        if !keep {
            let _ = fs::remove_file(entry.path());
        }
    }
}
