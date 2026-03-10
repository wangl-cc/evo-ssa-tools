use super::codec::CodecEngine;
use crate::error as crate_error;

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
pub enum Error {
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

pub type Result<T, E = Error> = std::result::Result<T, E>;

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
    /// Fetch the encoded bytes for `key` and pass them to `f`.
    ///
    /// Implementers are responsible only for locating the encoded value by `key`
    /// and borrowing it for the duration of `f`.
    fn fetch_encoded_with<T, E, F>(&self, key: &[u8], f: F) -> std::result::Result<T, E>
    where
        F: FnOnce(Option<&[u8]>) -> std::result::Result<T, E>,
        E: From<Error>;

    /// Store an already-encoded payload for `key`.
    fn store_encoded(&self, key: &[u8], encoded: &[u8]) -> Result<()>;

    /// Attempts to fetch a value from the cache.
    fn fetch<T, CE>(&self, key: &[u8], engine: &mut CE) -> crate_error::Result<Option<T>>
    where
        CE: CodecEngine<T>,
    {
        self.fetch_encoded_with(key, |encoded| {
            Ok(encoded.map(|bytes| engine.decode(bytes)).transpose()?)
        })
    }

    /// Stores a value in the cache.
    fn store<T, CE>(&self, key: &[u8], engine: &mut CE, value: &T) -> crate_error::Result<()>
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

        self.store_encoded(key, encoded)
            .map_err(crate_error::Error::from)
    }

    /// Fetch value by key, or execute and store it on cache miss.
    fn fetch_or_execute<T, CE, F, E>(
        &self,
        key: &[u8],
        engine: &mut CE,
        execute: F,
    ) -> std::result::Result<T, E>
    where
        CE: CodecEngine<T>,
        F: FnOnce(&mut CE) -> std::result::Result<T, E>,
        E: From<crate_error::Error>,
    {
        if let Some(cached) = self.fetch::<T, CE>(key, engine).map_err(E::from)? {
            Ok(cached)
        } else {
            let output = execute(engine)?;
            self.store::<T, CE>(key, engine, &output).map_err(E::from)?;
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
    fn fetch_encoded_with<T, E, F>(&self, _: &[u8], f: F) -> std::result::Result<T, E>
    where
        F: FnOnce(Option<&[u8]>) -> std::result::Result<T, E>,
        E: From<Error>,
    {
        f(None)
    }

    fn store<T, CE>(&self, _: &[u8], _: &mut CE, _: &T) -> crate_error::Result<()>
    where
        CE: CodecEngine<T>,
    {
        Ok(())
    }

    fn store_encoded(&self, _: &[u8], _: &[u8]) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests;
