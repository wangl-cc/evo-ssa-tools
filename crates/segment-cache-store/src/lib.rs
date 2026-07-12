#![doc = include_str!("../README.md")]
//!
//! # Code layout
//!
//! The implementation is organized around persistent concepts and their
//! invariants rather than horizontal technical layers:
//!
//! 1. `record`: borrowed sorted key/value input shared by encoders and commits
//! 2. `value`: the store-wide fixed or variable value layout
//! 3. `block`: one physical read unit and its encoding, decoding, checksum, and compression rules
//! 4. `segment`: one immutable sorted file, including geometry, identity, codecs, IO, and the open
//!    handle
//! 5. `catalog`: caller metadata, namespace identity, manifest visibility, and the selected segment
//!    snapshot
//! 6. `snapshot`: point, ordered, and range reads over one catalog snapshot
//! 7. `commit`: validated batches, pure transition plans, and atomic publication
//! 8. [`Store`]: the cheaply cloneable public facade and shared synchronization state
//!
//! `binary`, `key`, and `error` are small supporting modules rather than
//! architectural layers.

#[cfg(not(unix))]
compile_error!("segment-cache-store currently supports Unix targets only");

mod binary;
mod block;
mod catalog;
mod commit;
mod error;
mod key;
mod record;
mod segment;
mod snapshot;
mod store;
mod value;

pub use block::{BlockChecksumKind, ValuePayloadCompressionKind};
#[cfg(feature = "value-compression")]
pub use block::{CompressionPolicyError, ValuePayloadCompressionPolicy};
pub use catalog::{
    CatalogError, CatalogMismatch, CreateOptions, ManifestEncodeError, ManifestParseError,
    MetadataParseError, OpenOptions, StoreFileParseError, StoreInfo, StoreMetadata,
    StoreStorageStats,
};
pub use commit::{CommitOptions, CommitStats, WriteBatch};
pub use error::{CorruptionError, Error, FormatError, InputError, OptionsError, Result};
pub use snapshot::{OrderedLookup, RangeCursor};
pub use store::Store;
pub use value::ValueLayout;
