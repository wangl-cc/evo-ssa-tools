//! Write path: buffered batches and replacing-manifest commits.
//!
//! Symmetric to [`crate::read`]. A [`WriteBatch`] is the in-memory arena of
//! records a caller stages; [`commit`] merges that batch with the intersecting
//! visible segments, writes replacement segment files through the engine's
//! atomic publication protocol, and swaps in a new `MANIFEST`. The write-side
//! methods of [`crate::Store`] (`begin_batch`, `commit_batch`, `flush`) live in
//! [`commit`].

mod batch;
mod commit;

pub use batch::WriteBatch;
pub use commit::{CommitOptions, CommitStats};
