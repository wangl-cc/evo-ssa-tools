//! Atomic manifest transitions from buffered batches and store merges.
//!
//! A [`WriteBatch`] is the in-memory arena of records a caller stages.
//! [`plan`] owns the pure replacement decision. [`publish`] writes disjoint
//! batches directly, uses patch segments for eligible overlapping batches, or
//! normalizes overlap before atomically swapping `MANIFEST`.
//! Cross-store merge logic lives in [`merge`].

use std::sync::Arc;

use crate::store::StoreInner;

mod batch;
mod merge;
mod options;
mod plan;
mod publish;

pub use batch::WriteBatch;
pub use options::CommitOptions;
pub use publish::CommitStats;

pub(crate) struct Committer<'a> {
    inner: &'a Arc<StoreInner>,
}

impl<'a> Committer<'a> {
    pub(crate) fn new(inner: &'a Arc<StoreInner>) -> Self {
        Self { inner }
    }
}
