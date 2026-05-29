//! Storage providers and raw stores for `ssa-workflow`.
//!
//! All stores implement the raw `key -> encoded bytes` contract exposed by [`EncodedStorage`].
//! The higher-level task and transform types are responsible for canonical input encoding and value
//! serialization; storage providers decide where those bytes live and how they are shared.
//!
//! Provider selection guidance:
//!
//! - `Fjall3Store`: raw Fjall v3 keyspace-backed storage, opened manually by namespace name when
//!   the `fjall3` feature is enabled.
//! - `Fjall3StorageProvider`: recipe that opens named keyspaces in an existing Fjall v3 database
//!   with shared keyspace creation options when the `fjall3` feature is enabled.
//!
//! Stores do not add namespacing on top of the underlying database. Reuse the same partition,
//! keyspace, or table only when the cached compute semantics are intentionally identical.

mod namespace;
pub use namespace::StorageNamespace;

#[cfg(feature = "fjall3")]
mod fjall3;
#[cfg(feature = "fjall3")]
pub use fjall3::{Fjall3StorageProvider, Fjall3Store};

/// Errors produced by storage backends.
#[derive(thiserror::Error, Debug)]
pub enum StorageError {
    #[cfg(feature = "fjall3")]
    #[error("Fjall v3 database error")]
    Fjall3(#[from] ::fjall3::Error),
}

/// Result type returned by storage backends.
pub type StorageResult<T, E = StorageError> = std::result::Result<T, E>;

pub(crate) mod sealed {
    pub trait Sealed {}

    #[cfg(test)]
    impl<T: Sync> Sealed for T {}
}

/// Encoded storage for memoized `key -> encoded value` entries.
///
/// Keys are opaque bytes produced by canonical input encoding
/// ([`CanonicalEncode`](crate::cache::CanonicalEncode)).
/// Values are encoded byte payloads managed by the configured
/// [`CodecEngine`](crate::cache::codec::CodecEngine).
///
/// This trait is `Sync` because stores are shared across parallel workers.
/// Implementations are expected to be thread-safe for concurrent reads and writes.
///
/// This trait deliberately does not define any keyspace or schema management. If multiple compute
/// nodes share the same underlying backing store, the caller must ensure they also share identical
/// cache semantics.
///
/// This trait is sealed and can only be implemented by crate-provided storage backends.
pub trait EncodedStorage: sealed::Sealed + Sync {
    /// Borrowed view of an encoded value returned by this store.
    type Encoded<'a>: AsRef<[u8]>
    where
        Self: 'a;

    /// Fetch the encoded bytes for `key`.
    fn fetch_encoded(&self, key: &[u8]) -> StorageResult<Option<Self::Encoded<'_>>>;

    /// Store an already-encoded payload for `key`.
    fn store_encoded(&self, key: &[u8], encoded: &[u8]) -> StorageResult<()>;
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests;
