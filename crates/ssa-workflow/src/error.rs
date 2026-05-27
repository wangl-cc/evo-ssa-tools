//! Error and result types used throughout `ssa-workflow`.
//!
//! In most APIs, errors are surfaced as [`Error`], and the crate uses [`Result`] as a convenience
//! alias.

/// Error type for execution, caching, and serialization/deserialization.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    Codec(#[from] crate::cache::codec::Error),

    #[error(transparent)]
    Storage(#[from] crate::cache::storage::StorageError),

    /// A single-space managed memory cache was rebound to a different computation path.
    #[error("managed cache is already bound to {existing}, cannot bind it to {requested}")]
    ManagedCacheAlreadyBound { existing: String, requested: String },

    /// Execution was short-circuited due to an interrupt signal.
    #[error("Execution interrupted")]
    Interrupted,

    /// User compute function returned an error.
    #[error("Compute error")]
    Compute(#[from] Box<dyn std::error::Error + Send + Sync>),
}

/// Convenience alias for `std::result::Result<T, Error>`.
pub type Result<T, E = Error> = std::result::Result<T, E>;
