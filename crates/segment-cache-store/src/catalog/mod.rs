//! Store catalog: namespace identity, visibility, lifecycle, and maintenance.
//!
//! `STORE` defines persistent namespace identity and `MANIFEST` defines the
//! visible immutable segment set. This module owns their paths, parsing,
//! atomic publication, opening, creation, explicit garbage collection, and
//! physical storage statistics.

mod create;
mod descriptor;
mod error;
mod gc;
mod io;
mod manifest;
mod metadata;
mod open;
mod paths;
mod storage;
mod visibility;

pub use create::CreateOptions;
pub use descriptor::StoreFileParseError;
pub use error::{CatalogError, CatalogMismatch};
pub(crate) use io::WriterLock;
pub use manifest::{ManifestEncodeError, ManifestParseError};
pub(crate) use manifest::{SegmentManifestEntry, SegmentTier, StoreManifest};
pub use metadata::{MetadataParseError, StoreMetadata};
pub(crate) use open::OpenedStore;
pub use open::{OpenOptions, StoreInfo};
pub(crate) use paths::StorePaths;
pub use storage::StoreStorageStats;
pub(crate) use visibility::{SegmentSnapshot, VisibleSnapshot};

pub(crate) struct Catalog {
    paths: StorePaths,
}

impl Catalog {
    pub(crate) fn at(root: impl AsRef<std::path::Path>) -> Self {
        Self {
            paths: StorePaths::new(root),
        }
    }
}
