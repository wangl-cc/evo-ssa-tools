//! Error and result types used throughout `ssa-cache`.
//!
//! In most APIs, errors are surfaced as [`Error`], and the crate uses [`Result`] as a convenience
//! alias.

/// Error type for execution, caching, and serialization/deserialization.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    Codec(#[from] crate::cache::codec::Error),

    #[error(transparent)]
    Storage(#[from] crate::cache::storage::Error),

    /// Cache index was out of bounds when selecting a cache from a collection of caches.
    #[error("Try to get cache #{want} but only {total} available")]
    CacheOutofIndex { total: usize, want: usize },

    /// Execution was short-circuited due to an interrupt signal.
    #[error("Execution interrupted")]
    Interrupted,

    /// User compute function returned an error.
    #[error("Compute error")]
    Compute(#[from] Box<dyn std::error::Error + Send + Sync>),
}

/// Convenience alias for `std::result::Result<T, Error>`.
pub type Result<T, E = Error> = std::result::Result<T, E>;
