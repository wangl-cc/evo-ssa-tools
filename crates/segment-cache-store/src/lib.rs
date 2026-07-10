#![doc = include_str!("../README.md")]
//!
//! # Code layout
//!
//! The crate is layered so each level can be reviewed against `docs/design.md`
//! mostly on its own. The implementation layers form an acyclic chain —
//! `format ← engine ← read`, with `write` building on `engine` and `read` — and
//! [`Store`] sits on top as the facade: `store` collects the whole operational
//! API as thin delegators into `read` and `write`. `Store` is also the shared
//! handle those layers hold, so it is a deliberate hub rather than another link
//! in the chain; the heavy code never depends back up onto the facade's API.
//!
//! 1. `error`: the public [`Error`] and the input-contract enums, plus every conversion into
//!    [`Error`]. Each layer owns its own precise error types (e.g. `format` owns
//!    [`CatalogError`]/[`FormatError`]/[`CorruptionError`]); this module is where they aggregate at
//!    the API boundary.
//! 2. `format`: pure byte layouts — no filesystem access
//! 3. `engine`: open-store substrate — filesystem primitives, runtime state, create/open, explicit
//!    GC
//! 4. `read`: point lookups and range cursors over engine snapshots
//! 5. `write`: buffered batches and replacing-manifest commits
//! 6. [`Store`]: the cheaply cloneable public handle, whose operational API is assembled in `store`
//!    from the read and write layers

mod engine;
mod error;
mod format;
mod read;
mod store;
mod write;

pub use engine::{CreateOptions, OpenOptions, StoreInfo, StoreStorageStats};
pub use error::{Error, InputError, OptionsError, Result};
pub use format::{
    BlockChecksumKind, CatalogError, CatalogMismatch, CompressionPolicyError, CorruptionError,
    FormatError, ManifestEncodeError, ManifestParseError, MetadataParseError, StoreFileParseError,
    StoreMetadata, ValueLayout, ValuePayloadCompressionKind, ValuePayloadCompressionPolicy,
};
pub use read::{OrderedLookup, RangeCursor};
pub use store::Store;
pub use write::{CommitOptions, CommitStats, WriteBatch};
