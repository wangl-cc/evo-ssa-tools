//! Store catalog: namespace identity, visibility, lifecycle, and maintenance.
//!
//! `STORE` defines persistent namespace identity and `MANIFEST` defines the
//! visible immutable segment set. This module owns their paths, parsing,
//! atomic publication, opening, creation, explicit garbage collection, and
//! physical storage statistics.

mod create;
pub(crate) mod descriptor;
pub(crate) mod gc;
pub(crate) mod io;
pub(crate) mod manifest;
mod open;
pub(crate) mod paths;
pub(crate) mod storage;

pub use create::CreateOptions;
pub use descriptor::StoreFileParseError;
pub use manifest::{ManifestEncodeError, ManifestParseError};
pub use open::{OpenOptions, StoreInfo};
pub use storage::StoreStorageStats;
