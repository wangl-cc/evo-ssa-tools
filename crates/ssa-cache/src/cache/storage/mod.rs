use super::codec::CodecEngine;
use crate::error::Result;

mod hashmap;
pub use hashmap::{DefaultHashMapStore, HashMapStore};

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

pub type StorageResult<T, E = StorageError> = std::result::Result<T, E>;

#[doc(hidden)]
pub trait WorkerForkStore: private::Sealed + Sync {
    fn fork_store(&self) -> Self;
}

mod private {
    pub trait Sealed {}
}

/// Storage backend for `key -> value` entries.
///
/// Keys are opaque bytes produced by canonical input encoding
/// ([`CanonicalEncode`](crate::cache::canonical_encode::CanonicalEncode)).
/// Values are encoded byte payloads managed by the configured [`CodecEngine`].
///
/// This trait is `Sync` because stores are shared across parallel workers.
/// Implementations are expected to be thread-safe for concurrent reads and writes.
///
/// `()` implements `CacheStore` as a "no-cache" backend that always misses and discards writes.
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
            Some(encoded) => Ok(Some(engine.decode(encoded.as_ref())?)),
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
                eprintln!("[ssa-cache] skipping cache write: {reason}");
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

impl private::Sealed for () {}

impl WorkerForkStore for () {
    fn fork_store(&self) -> Self {
        *self
    }
}

impl CacheStore for () {
    type Encoded<'a>
        = &'a [u8]
    where
        Self: 'a;

    fn fetch_encoded(&self, _: &[u8]) -> StorageResult<Option<Self::Encoded<'_>>> {
        Ok(None)
    }

    fn store<T, CE>(&self, _: &[u8], _: &mut CE, _: &T) -> Result<()>
    where
        CE: CodecEngine<T>,
    {
        Ok(())
    }

    fn store_encoded(&self, _: &[u8], _: &[u8]) -> StorageResult<()> {
        Ok(())
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests;
