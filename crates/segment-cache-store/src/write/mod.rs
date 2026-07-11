//! Write path: buffered batches and replacing-manifest commits.
//!
//! Symmetric to [`crate::read`]. A [`WriteBatch`] is the in-memory arena of
//! records a caller stages. [`commit`] publishes disjoint batches directly,
//! uses patch segments for eligible overlapping batches, or normalizes overlap
//! into replacement main segments before atomically swapping `MANIFEST`.
//! Cross-store merge logic lives in [`merge`].

mod batch;
mod commit;
mod merge;

pub use batch::WriteBatch;
pub use commit::{CommitOptions, CommitStats};
