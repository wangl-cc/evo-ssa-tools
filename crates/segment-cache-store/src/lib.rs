#![doc = include_str!("../README.md")]
//!
//! # Code layout
//!
//! The implementation is organized around persistent concepts and their
//! invariants rather than horizontal technical layers:
//!
//! 1. `block`: one physical read unit and its encoding, decoding, checksum, and compression rules
//! 2. `schema`: persistent store geometry and caller-defined namespace metadata
//! 3. `segment`: one immutable sorted file, including records, format, file access, and runtime
//!    state
//! 4. `catalog`: namespace identity and the manifest that selects visible segments
//! 5. `snapshot`: point, ordered, and range reads over one visible segment snapshot
//! 6. `commit`: validated batches and atomic transitions to a new visible snapshot
//! 7. [`Store`]: the cheaply cloneable public facade and its shared state
//!
//! `binary`, `key`, and `error` are small supporting modules rather than
//! architectural layers.

#[cfg(not(unix))]
compile_error!("segment-cache-store currently supports Unix targets only");

mod binary;
pub(crate) mod block;
mod catalog;
mod commit;
mod error;
mod key;
mod schema;
pub(crate) mod segment;
mod snapshot;
mod store;

pub use block::{
    BlockChecksumKind, CompressionPolicyError, ValuePayloadCompressionKind,
    ValuePayloadCompressionPolicy,
};
pub use catalog::{
    CreateOptions, ManifestEncodeError, ManifestParseError, OpenOptions, StoreFileParseError,
    StoreInfo, StoreStorageStats,
};
pub use commit::{CommitOptions, CommitStats, WriteBatch};
pub use error::{
    CatalogError, CatalogMismatch, CorruptionError, Error, FormatError, InputError, OptionsError,
    Result,
};
pub use schema::{MetadataParseError, StoreMetadata, ValueLayout};
pub use snapshot::{OrderedLookup, RangeCursor};
pub use store::Store;
