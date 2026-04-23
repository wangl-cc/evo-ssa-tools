//! Storage backends for `ssa-pipeline`.
//!
//! All stores implement the same raw `key -> encoded bytes` contract exposed by [`CacheStore`].
//! The higher-level step and pipeline types are responsible for canonical input encoding and value
//! serialization; storage backends only decide where those bytes live and how they are shared.
//!
//! Backend selection guidance:
//!
//! - [`Fjall2Store`]: persistent Fjall v2 partition-backed storage when you already manage a
//!   [`fjall::Keyspace`](::fjall2::Keyspace) externally.
//! - [`Fjall3Store`]: persistent Fjall v3 keyspace-backed storage when you already manage a
//!   [`fjall::Database`](::fjall3::Database) externally.
//! - [`RedbStore`]: persistent single-file storage scoped to a [`redb::Database`](::redb::Database)
//!   table.
//!
//! Stores do not add namespacing on top of the underlying database. Reuse the same partition,
//! keyspace, or table only when the cached compute semantics are intentionally identical.

use super::codec::CodecEngine;
use crate::error::Result;

#[cfg(feature = "fjall2")]
mod fjall2;
#[cfg(feature = "fjall2")]
pub use fjall2::Fjall2Store;

#[cfg(feature = "fjall3")]
mod fjall3;
#[cfg(feature = "fjall3")]
pub use fjall3::Fjall3Store;

#[cfg(feature = "redb")]
mod redb;
#[cfg(feature = "redb")]
pub use redb::RedbStore;

/// Errors produced by storage backends.
#[derive(thiserror::Error, Debug)]
pub enum StorageError {
    #[cfg(feature = "fjall2")]
    #[error("Fjall v2 database error")]
    Fjall2(#[from] ::fjall2::Error),

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

    /// Attempts to fetch a value from the cache.
    fn fetch<T, CE>(&self, key: &[u8], engine: &mut CE) -> Result<Option<T>>
    where
        CE: CodecEngine<T>,
    {
        match self.fetch_encoded(key)? {
            Some(encoded) => match engine.decode(encoded.as_ref()) {
                Ok(value) => Ok(Some(value)),
                Err(error) if error.is_cache_corruption() => {
                    eprintln!("[ssa-pipeline] ignoring corrupted cache entry during read: {error}");
                    Ok(None)
                }
                Err(error) => Err(error.into()),
            },
            None => Ok(None),
        }
    }

    /// Stores a value in the cache.
    fn store<T, CE>(&self, key: &[u8], engine: &mut CE, value: &T) -> Result<()>
    where
        CE: CodecEngine<T>,
    {
        let encoded = match engine.encode(value) {
            Ok(encoded) => encoded,
            Err(reason) => {
                eprintln!("[ssa-pipeline] skipping cache write: {reason}");
                return Ok(());
            }
        };

        self.store_encoded(key, encoded)?;
        Ok(())
    }

    /// Fetch value by key, or execute and store it on cache miss.
    fn fetch_or_execute<T, CE, F>(&self, key: &[u8], engine: &mut CE, execute: F) -> Result<T>
    where
        CE: CodecEngine<T>,
        F: FnOnce(&mut CE) -> Result<T>,
    {
        if let Some(cached) = self.fetch::<T, CE>(key, engine)? {
            Ok(cached)
        } else {
            let output = execute(engine)?;
            self.store::<T, CE>(key, engine, &output)?;
            Ok(output)
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests;
