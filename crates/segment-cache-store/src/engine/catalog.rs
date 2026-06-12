//! Catalog file IO: loading and atomically publishing `STORE` and `MANIFEST`.
//!
//! Byte layouts and structural validation live in [`crate::format`]; this
//! module only moves those bytes to and from disk.

use std::fs;

use crate::{
    engine::paths::StorePaths,
    error::Result,
    format::{manifest::StoreManifest, store_file::StoreDescriptor},
};

pub(crate) fn load_descriptor(paths: &StorePaths) -> Result<Option<StoreDescriptor>> {
    let path = paths.store_file();
    if !path.exists() {
        return Ok(None);
    }
    Ok(Some(StoreDescriptor::parse(&fs::read_to_string(path)?)?))
}

pub(crate) fn publish_descriptor(paths: &StorePaths, descriptor: &StoreDescriptor) -> Result<()> {
    paths
        .store_file_publish()
        .write_bytes(descriptor.encode().as_bytes())
}

pub(crate) fn load_manifest(paths: &StorePaths) -> Result<Option<StoreManifest>> {
    let path = paths.manifest();
    if !path.exists() {
        return Ok(None);
    }
    Ok(Some(StoreManifest::parse(&fs::read(path)?)?))
}

pub(crate) fn publish_manifest(paths: &StorePaths, manifest: &StoreManifest) -> Result<()> {
    paths.manifest_publish().write_bytes(&manifest.encode()?)
}
