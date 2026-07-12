//! Atomic manifest transitions from buffered batches and store merges.
//!
//! A [`WriteBatch`] is the in-memory arena of records a caller stages.
//! [`execution`] publishes disjoint batches directly,
//! uses patch segments for eligible overlapping batches, or normalizes overlap
//! into replacement main segments before atomically swapping `MANIFEST`.
//! Cross-store merge logic lives in [`merge`].

pub(crate) mod batch;
pub(crate) mod execution;
pub(crate) mod merge;
pub(crate) mod options;

pub use batch::WriteBatch;
pub use execution::CommitStats;
pub use options::CommitOptions;
