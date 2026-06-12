//! Filesystem engine: every byte that reaches or leaves disk goes through
//! this tree.
//!
//! The engine owns paths, atomic publication, the advisory writer lock,
//! runtime state, segment file access, store lifecycle (create/open/commit),
//! and garbage collection. It composes the pure encoders and decoders from
//! [`crate::format`]; it never re-implements byte layouts.
//!
//! Review topology:
//!
//! - [`io`]: atomic publish protocol, writer lock, positioned reads
//! - [`paths`] and [`catalog`]: where files live and how catalog files are
//!   loaded and published
//! - [`segment_file`]: opening and validating one segment file, block reads
//! - [`runtime`]: in-memory shared state behind the public `Store` handle
//! - [`create`], [`open`], [`commit`], [`gc`]: store lifecycle operations

pub(crate) mod catalog;
mod commit;
mod create;
mod gc;
pub(crate) mod io;
mod open;
pub(crate) mod paths;
pub(crate) mod runtime;
pub(crate) mod segment_file;

pub use commit::{CommitOptions, CommitStats};
pub use create::CreateOptions;
pub use open::OpenOptions;
