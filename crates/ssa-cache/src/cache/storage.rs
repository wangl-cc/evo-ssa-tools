use std::sync::{Arc, RwLock};

use crate::{Result, cache::codec::CodecEngine};

/// Storage backend for `key -> value` entries.
///
/// Keys are opaque bytes produced by canonical input encoding
/// ([`CanonicalEncode`](crate::cache::canonical_encode::CanonicalEncode)).
/// Values are encoded byte payloads managed by the configured [`EngineCodec`].
///
/// This trait is `Sync` because stores are shared across parallel workers.
/// Implementations are expected to be thread-safe for concurrent reads and writes.
///
/// `()` implements `CacheStore` as a "no-cache" backend that always misses and discards writes.
pub trait CacheStore: Sync {
    /// Attempts to fetch a value from the cache.
    ///
    /// Returns `Ok(Some(value))` on hit, `Ok(None)` on miss.
    fn fetch<T, E>(&self, key: &[u8], engine: &mut E) -> Result<Option<T>>
    where
        E: CodecEngine<T>;

    /// Stores a value in the cache.
    fn store<T, E>(&self, key: &[u8], engine: &mut E, value: &T) -> Result<()>
    where
        E: CodecEngine<T>;

    /// Fetch value by key, or execute and store it on cache miss.
    fn fetch_or_execute<T, E, F>(&self, key: &[u8], engine: &mut E, execute: F) -> Result<T>
    where
        E: CodecEngine<T>,
        F: FnOnce(&mut E) -> Result<T>,
    {
        if let Some(cached) = self.fetch::<T, E>(key, engine)? {
            Ok(cached)
        } else {
            let output = execute(engine)?;
            self.store::<T, E>(key, engine, &output)?;
            Ok(output)
        }
    }
}

impl CacheStore for () {
    fn fetch<T, E>(&self, _: &[u8], _: &mut E) -> Result<Option<T>>
    where
        E: CodecEngine<T>,
    {
        Ok(None)
    }

    fn store<T, E>(&self, _: &[u8], _: &mut E, _: &T) -> Result<()>
    where
        E: CodecEngine<T>,
    {
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
    fn fetch<T, E>(&self, key: &[u8], engine: &mut E) -> Result<Option<T>>
    where
        E: CodecEngine<T>,
    {
        let map = self
            .0
            .read()
            .expect("cache read lock poisoned: HashMapStore should not panic while holding lock");
        map.get(key)
            .map(|v| engine.decode(v.as_slice()))
            .transpose()
    }

    fn store<T, E>(&self, key: &[u8], engine: &mut E, value: &T) -> Result<()>
    where
        E: CodecEngine<T>,
    {
        let encoded = engine.encode(value).to_vec();
        let mut map = self
            .0
            .write()
            .expect("cache write lock poisoned: HashMapStore should not panic while holding lock");
        map.insert(key.to_owned(), encoded);
        Ok(())
    }
}

#[cfg(feature = "fjall")]
mod fjall_store {
    use super::*;

    impl CacheStore for fjall::Keyspace {
        fn fetch<T, E>(&self, key: &[u8], engine: &mut E) -> Result<Option<T>>
        where
            E: CodecEngine<T>,
        {
            self.get(key)?
                .map(|v| engine.decode(v.as_ref()))
                .transpose()
        }

        fn store<T, E>(&self, key: &[u8], engine: &mut E, value: &T) -> Result<()>
        where
            E: CodecEngine<T>,
        {
            self.insert(key, engine.encode(value))?;
            Ok(())
        }
    }
}

#[cfg(test)]
#[cfg(feature = "bitcode")]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::cache::codec::bitcode::Bitcode;

    #[test]
    fn test_hashmap_store() -> Result<()> {
        let store = DefaultHashMapStore::default();
        let mut engine = Bitcode::default();

        let value = 42u32;
        let key = b"test_sig";
        store.store::<u32, Bitcode>(key, &mut engine, &value)?;
        assert_eq!(store.fetch::<u32, Bitcode>(key, &mut engine)?, Some(value));

        assert_eq!(
            store.fetch::<u32, Bitcode>(b"non_existent", &mut engine)?,
            None
        );

        assert!(matches!(
            store.fetch::<u64, Bitcode>(key, &mut engine),
            Err(crate::Error::BitCode(_))
        ));

        Ok(())
    }

    #[cfg(all(feature = "lz4", feature = "bitcode"))]
    #[test]
    fn test_hashmap_store_compressed_value_roundtrip() -> Result<()> {
        use crate::cache::codec::compress::{CompressedCodec, lz4::Lz4};
        type Lz4Engine = CompressedCodec<Bitcode, Lz4>;

        let store = DefaultHashMapStore::default();
        let mut engine = Lz4Engine::default();
        let key = b"compressible";
        let value = "a".repeat(96 * 1024);

        store.store::<String, Lz4Engine>(key, &mut engine, &value)?;
        assert_eq!(
            store.fetch::<String, Lz4Engine>(key, &mut engine)?,
            Some(value)
        );

        let map = store.0.read().unwrap();
        let encoded = map.get(key.as_slice()).expect("value should be stored");
        assert!(!encoded.is_empty());
        assert_eq!(encoded[0] & 0b1111_0000, 0b0001_0000);
        assert_eq!(encoded[0] & 0b0000_1111, 0b0001);
        Ok(())
    }

    #[cfg(feature = "fjall")]
    #[test]
    fn test_fjall_store() -> Result<()> {
        let tmp = tempfile::tempdir().unwrap();
        let db = fjall::Database::builder(&tmp).open()?;
        let partition = db.keyspace("test", Default::default)?;
        let mut engine = Bitcode::default();

        let value = 42u32;
        let key = b"test_sig";
        partition.store::<u32, Bitcode>(key, &mut engine, &value)?;
        assert_eq!(
            partition.fetch::<u32, Bitcode>(key, &mut engine)?,
            Some(value)
        );

        assert_eq!(
            partition.fetch::<u32, Bitcode>(b"non_existent", &mut engine)?,
            None
        );

        assert!(matches!(
            partition.fetch::<u64, Bitcode>(key, &mut engine),
            Err(crate::Error::BitCode(_))
        ));

        Ok(())
    }
}
