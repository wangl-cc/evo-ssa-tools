//! Open-store substrate: the filesystem mechanics, the in-memory runtime
//! state, and store lifecycle that the read and write paths sit on.
//!
//! Three sub-concerns, bottom to top: the filesystem primitives (paths, atomic
//! publication, the advisory writer lock, segment file access) through which
//! every byte to or from disk passes; the in-memory runtime state behind the
//! public `Store` handle; and the lifecycle operations (create, open, GC) that
//! build and maintain it. It composes the pure encoders and decoders from
//! [`crate::format`] and never re-implements byte layouts. The commit path
//! lives in [`crate::write`], not here.
//!
//! Review topology:
//!
//! - [`io`]: atomic publish protocol, writer lock, positioned reads
//! - [`paths`]: where files live; catalog file loading and publication
//! - [`segment_file`]: opening and validating one segment file, block reads
//! - [`runtime`]: in-memory shared state behind the public `Store` handle
//! - [`create`], [`open`], [`gc`]: store lifecycle operations
//!
//! The two data paths sit on top of this substrate as their own layers:
//! reads in [`crate::read`], batched writes in [`crate::write`].

mod create;
pub(crate) mod gc;
pub(crate) mod io;
mod open;
pub(crate) mod paths;
pub(crate) mod runtime;
pub(crate) mod segment_file;
pub(crate) mod storage;

pub use create::CreateOptions;
pub use open::{OpenOptions, StoreInfo};
pub use storage::StoreStorageStats;
