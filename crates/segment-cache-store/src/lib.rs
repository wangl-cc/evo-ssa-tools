#![doc = include_str!("../README.md")]
//!
//! # Code layout
//!
//! The crate is layered so each level can be reviewed against
//! `docs/design.md` independently, depending only on the levels above it:
//!
//! 1. `error`: the whole error model in one place
//! 2. `format`: pure byte layouts — no filesystem access
//! 3. `engine`: all filesystem mechanics — paths, atomic publication, the
//!    writer lock, runtime state, create/open/commit, GC
//! 4. `read`: ordered lookup sessions and range cursors over engine snapshots
//! 5. [`Store`] and [`WriteBatch`]: the public facade

mod batch;
mod engine;
mod error;
mod format;
mod read;
mod store;

pub use batch::WriteBatch;
pub use engine::{CommitOptions, CommitStats, CreateOptions, OpenOptions};
pub use error::{Error, InputError, OptionsError, Result};
pub use format::{
    CatalogError, CatalogMismatch, CorruptionError, FormatError, StoreMetadata, ValueLayout,
};
pub use read::{OrderedLookup, RangeCursor};
pub use store::Store;
