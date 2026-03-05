use std::sync::{Arc, RwLock};

use crate::{Result, cache::codec::Codec};

/// Storage backend for `key -> value` entries.
///
/// Keys are opaque bytes. In `ssa-cache`, keys are produced by canonical input encoding
/// ([`CanonicalEncode`](crate::cache::canonical_encode::CanonicalEncode)).
///
/// Values are encoded byte payloads managed by the configured
/// [`Codec`](crate::cache::codec::Codec).
///
/// This trait is `Sync` because stores are shared across parallel workers. Implementations are
/// expected to be thread-safe for concurrent reads and writes.
///
/// `()` implements `CacheStore` as a "no-cache" backend that always misses and discards writes.
pub trait CacheStore: Sync {
    /// Attempts to fetch a value from the cache.
    ///
    /// Returns `Ok(Some(value))` on hit, `Ok(None)` on miss.
    fn fetch<T: Codec>(&self, key: &[u8], buffer: &mut T::Buffer) -> Result<Option<T>>;

    /// Stores a value in the cache.
    fn store<T: Codec>(&self, key: &[u8], buffer: &mut T::Buffer, value: &T) -> Result<()>;

    /// Fetch value by key, or execute and store it on cache miss.
    ///
    /// The same `buffer` is reused for fetch, execute, and store to minimize allocations.
    fn fetch_or_execute<T, F>(&self, key: &[u8], buffer: &mut T::Buffer, execute: F) -> Result<T>
    where
        T: Codec,
        F: FnOnce(&mut T::Buffer) -> Result<T>,
    {
        if let Some(cached) = self.fetch(key, buffer)? {
            Ok(cached)
        } else {
            let output = execute(buffer)?;
            self.store(key, buffer, &output)?;
            Ok(output)
        }
    }
}

impl CacheStore for () {
    fn fetch<T: Codec>(&self, _: &[u8], _: &mut T::Buffer) -> Result<Option<T>> {
        Ok(None)
    }

    fn store<T: Codec>(&self, _: &[u8], _: &mut T::Buffer, _: &T) -> Result<()> {
        Ok(())
    }
}

type HashMap<H> = std::collections::HashMap<Vec<u8>, Vec<u8>, H>;

/// A simple in-memory cache store backed by a `HashMap`.
///
/// Clone is cheap and shares the underlying map (`Arc<RwLock<...>>`).
/// You may want a faster hasher like [`rapidhash`] instead of the default for better performance.
#[derive(Debug, Clone)]
pub struct HashMapStore<H>(Arc<RwLock<HashMap<H>>>);

impl<H> Default for HashMapStore<H>
where
    H: std::hash::BuildHasher + Default,
{
    fn default() -> Self {
        Self(Arc::new(RwLock::new(HashMap::<H>::default())))
    }
}

/// Type alias for a [`HashMapStore`] using the default hasher from [`std::collections::hash_map`].
pub type DefaultHashMapStore = HashMapStore<std::collections::hash_map::RandomState>;

impl<H> CacheStore for HashMapStore<H>
where
    H: std::hash::BuildHasher + Send + Sync,
{
    fn fetch<T: Codec>(&self, key: &[u8], buffer: &mut T::Buffer) -> Result<Option<T>> {
        let map = self
            .0
            .read()
            .expect("cache read lock poisoned: HashMapStore should not panic while holding lock");
        map.get(key)
            .map(|value| T::decode(value.as_slice(), buffer))
            .transpose()
    }

    fn store<T: Codec>(&self, key: &[u8], buffer: &mut T::Buffer, value: &T) -> Result<()> {
        let value = value.encode(buffer);
        let mut map = self
            .0
            .write()
            .expect("cache write lock poisoned: HashMapStore should not panic while holding lock");
        map.insert(key.to_owned(), value.to_vec());
        Ok(())
    }
}

#[cfg(feature = "fjall")]
mod fjall_store {
    use super::*;

    impl CacheStore for fjall::Keyspace {
        fn fetch<T: Codec>(&self, key: &[u8], buffer: &mut T::Buffer) -> Result<Option<T>> {
            let value = self.get(key)?;
            value
                .map(|value| T::decode(value.as_ref(), buffer))
                .transpose()
        }

        fn store<T: Codec>(&self, key: &[u8], buffer: &mut T::Buffer, value: &T) -> Result<()> {
            let value = value.encode(buffer);
            self.insert(key, value)?;
            Ok(())
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::cache::codec::CodecBuffer;

    #[test]
    fn test_hashmap_store() -> Result<()> {
        let store = DefaultHashMapStore::default();
        let mut buffer = <u32 as Codec>::Buffer::init();

        // Test storing and fetching a value
        let value = 42u32;
        let key = b"test_sig";
        store.store(key, &mut buffer, &value)?;
        assert_eq!(store.fetch(key, &mut buffer)?, Some(value));

        // Test fetching non-existent key
        assert_eq!(store.fetch::<u32>(b"non_existent", &mut buffer)?, None);

        // Test fetching key with wrong type
        assert!(matches!(
            store.fetch::<u64>(key, &mut buffer),
            Err(crate::Error::BitCode(_))
        ));

        Ok(())
    }

    #[cfg(feature = "fjall")]
    #[test]
    fn test_fjall_store() -> Result<()> {
        let tmp = tempfile::tempdir().unwrap();
        let db = fjall::Database::builder(&tmp).open()?;
        let partition = db.keyspace("test", Default::default)?;
        let mut buffer = <u32 as Codec>::Buffer::init();

        // Test storing and fetching a value
        let value = 42u32;
        let key = b"test_sig";
        partition.store(key, &mut buffer, &value)?;
        assert_eq!(partition.fetch(key, &mut buffer)?, Some(value));

        // Test fetching non-existent key
        assert_eq!(partition.fetch::<u32>(b"non_existent", &mut buffer)?, None);

        // Test fetching key with wrong type
        assert!(matches!(
            partition.fetch::<u64>(key, &mut buffer),
            Err(crate::Error::BitCode(_))
        ));

        Ok(())
    }
}
