//! Storage backends for `ssa-workflow`.
//!
//! All stores implement the same raw `key -> encoded bytes` contract exposed by [`CacheStore`].
//! The higher-level task and transform types are responsible for canonical input encoding and value
//! serialization; storage backends only decide where those bytes live and how they are shared.
//!
//! Backend selection guidance:
//!
//! - [`Fjall3Store`]: persistent Fjall v3 keyspace-backed storage when you already manage a
//!   [`fjall::Database`](::fjall3::Database) externally.
//! - [`RedbStore`]: persistent single-file storage scoped to a [`redb::Database`](::redb::Database)
//!   table.
//!
//! Stores do not add namespacing on top of the underlying database. Reuse the same partition,
//! keyspace, or table only when the cached compute semantics are intentionally identical.

mod namespace;
pub use namespace::StorageNamespace;

#[cfg(feature = "fjall3")]
mod fjall3;
#[cfg(feature = "fjall3")]
pub use fjall3::{Fjall3Backend, Fjall3Store};

#[cfg(feature = "redb")]
mod redb;
#[cfg(feature = "redb")]
pub use redb::{RedbBackend, RedbStore};

/// Errors produced by storage backends.
#[derive(thiserror::Error, Debug)]
pub enum StorageError {
    #[cfg(feature = "fjall3")]
    #[error("Fjall v3 database error")]
    Fjall3(#[from] ::fjall3::Error),

    #[cfg(feature = "redb")]
    #[error("redb database error")]
    Redb(#[from] ::redb::Error),
}

/// Result type returned by storage backends.
pub type StorageResult<T, E = StorageError> = std::result::Result<T, E>;

/// Storage backend for memoized `key -> value` entries.
///
/// Keys are opaque bytes produced by canonical input encoding
/// ([`CanonicalEncode`](crate::cache::CanonicalEncode)).
/// Values are encoded byte payloads managed by the configured [`CodecEngine`].
///
/// This trait is `Sync` because stores are shared across parallel workers.
/// Implementations are expected to be thread-safe for concurrent reads and writes.
///
/// This trait deliberately does not define any keyspace or schema management. If multiple compute
/// nodes share the same underlying backing store, the caller must ensure they also share identical
/// cache semantics.
pub trait CacheStore: Sync {
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
