/// Internal warning helper.
///
/// Uses [`log::warn`] when the `log` feature is enabled; falls back to [`eprintln`] otherwise.
#[cfg(feature = "log")]
macro_rules! warn {
    ($($arg:tt)*) => { log::warn!($($arg)*) };
}
#[cfg(not(feature = "log"))]
macro_rules! warn {
    ($($arg:tt)*) => { eprintln!($($arg)*) };
}
pub use encoded::EncodedCache;
#[cfg(feature = "lru")]
pub use memory::ManagedLruCache;
pub use memory::{ManagedHashCache, ManagedMemoryCache};
pub use provider::{CacheProvider, PersistentCacheProvider, StorageProvider, StorageProviderExt};
pub use ssa_canonical_key::{CanonicalBuffer, CanonicalEncode, CanonicalWriter};

use crate::Result;

/// Execution-facing cache abstraction keyed by canonical input bytes.
pub trait Cache<T> {
    /// Fetch a typed value by key. Returns `Ok(None)` on miss.
    fn fetch(&mut self, key: &[u8]) -> Result<Option<T>>;

    /// Store a typed value by key.
    fn store(&mut self, key: &[u8], value: &T) -> Result<()>;

    /// Fetch a typed value by key, or execute and store it on cache miss.
    ///
    /// This default implementation is thread-safe when the underlying cache is thread-safe, but it
    /// is not single-flight: concurrent misses for the same key may execute the closure more than
    /// once before any worker stores a value.
    fn fetch_or_execute<F>(&mut self, key: &[u8], execute: F) -> Result<T>
    where
        F: FnOnce() -> Result<T>,
    {
        if let Some(cached) = self.fetch(key)? {
            return Ok(cached);
        }
        let value = execute()?;
        self.store(key, &value)?;
        Ok(value)
    }
}

impl<T> Cache<T> for () {
    fn fetch(&mut self, _key: &[u8]) -> Result<Option<T>> {
        Ok(None)
    }

    fn store(&mut self, _key: &[u8], _value: &T) -> Result<()> {
        Ok(())
    }
}

/// Clone a new handle that shares the same backing state.
///
/// This is the contract for caches and raw stores whose worker-local instance should keep pointing
/// at the same underlying data. Cloning is shallow at the logical level: one worker can populate
/// the cache or store and other workers immediately observe the same entries.
///
/// Typical implementations wrap an `Arc` or another shared database handle.
pub trait CloneShared: Sized {
    fn clone_shared(&self) -> Self;
}

impl CloneShared for () {
    fn clone_shared(&self) -> Self {}
}

pub mod codec;
mod encoded;
pub mod memory;
mod provider;
pub mod storage;

#[cfg(feature = "migrate")]
pub mod migrate;

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    mod unit_cache {
        use crate::{
            Result,
            cache::{Cache, codec::CloneFresh},
        };

        #[test]
        fn executes_without_storage() -> Result<()> {
            let mut cache = ();
            let value = cache.fetch_or_execute(b"ignored", || Ok(11u32))?;
            assert_eq!(value, 11);
            Ok(())
        }

        #[test]
        fn unit_clone_fresh_is_noop() {
            ().clone_fresh();
        }
    }
}
