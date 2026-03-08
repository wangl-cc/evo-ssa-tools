use std::sync::{Arc, RwLock};

use crate::{Result, cache::codec::CodecEngine};

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
    ///
    /// - On hit, call `f(Some(bytes))`.
    /// - On miss, call `f(None)`.
    /// - The borrowed `bytes` only need to remain valid until `f` returns.
    fn fetch_encoded_with<T, F>(&self, key: &[u8], f: F) -> Result<T>
    where
        F: FnOnce(Option<&[u8]>) -> Result<T>;

    /// Store an already-encoded payload for `key`.
    fn store_encoded(&self, key: &[u8], encoded: &[u8]) -> Result<()>;

    /// Attempts to fetch a value from the cache.
    ///
    /// Returns `Ok(Some(value))` on hit, `Ok(None)` on miss.
    fn fetch<T, E>(&self, key: &[u8], engine: &mut E) -> Result<Option<T>>
    where
        E: CodecEngine<T>,
    {
        self.fetch_encoded_with(key, |encoded| {
            encoded.map(|bytes| engine.decode(bytes)).transpose()
        })
    }

    /// Stores a value in the cache.
    fn store<T, E>(&self, key: &[u8], engine: &mut E, value: &T) -> Result<()>
    where
        E: CodecEngine<T>,
    {
        let encoded = match engine.encode(value) {
            Ok(encoded) => encoded,
            Err(reason) => {
                eprintln!("[ssa-cache] skipping cache write: {reason}");
                return Ok(());
            }
        };
        self.store_encoded(key, encoded)
    }

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
    fn fetch_encoded_with<T, F>(&self, _: &[u8], f: F) -> Result<T>
    where
        F: FnOnce(Option<&[u8]>) -> Result<T>,
    {
        f(None)
    }

    fn store<T, E>(&self, _: &[u8], _: &mut E, _: &T) -> Result<()>
    where
        E: CodecEngine<T>,
    {
        Ok(())
    }

    fn store_encoded(&self, _: &[u8], _: &[u8]) -> Result<()> {
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
    fn fetch_encoded_with<T, F>(&self, key: &[u8], f: F) -> Result<T>
    where
        F: FnOnce(Option<&[u8]>) -> Result<T>,
    {
        let map = self
            .0
            .read()
            .expect("cache read lock poisoned: HashMapStore should not panic while holding lock");
        f(map.get(key).map(|bytes| bytes.as_slice()))
    }

    fn store_encoded(&self, key: &[u8], encoded: &[u8]) -> Result<()> {
        let mut map = self
            .0
            .write()
            .expect("cache write lock poisoned: HashMapStore should not panic while holding lock");
        map.insert(key.to_owned(), encoded.to_vec());
        Ok(())
    }
}

#[cfg(feature = "fjall")]
mod fjall_store {
    use super::*;

    impl CacheStore for fjall::Keyspace {
        fn fetch_encoded_with<T, F>(&self, key: &[u8], f: F) -> Result<T>
        where
            F: FnOnce(Option<&[u8]>) -> Result<T>,
        {
            let value = self.get(key)?;
            f(value.as_ref().map(|bytes| bytes.as_ref()))
        }

        fn store_encoded(&self, key: &[u8], encoded: &[u8]) -> Result<()> {
            self.insert(key, encoded)?;
            Ok(())
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::cache::codec::{CodecEngine, SkipReason, fixtures::FixtureEngine};

    #[test]
    fn test_hashmap_store() -> Result<()> {
        let store = DefaultHashMapStore::default();
        let mut engine = FixtureEngine::default();

        let value = 42u32;
        let key = b"test_sig";
        store.store::<u32, FixtureEngine>(key, &mut engine, &value)?;
        assert_eq!(
            store.fetch::<u32, FixtureEngine>(key, &mut engine)?,
            Some(value)
        );

        assert_eq!(
            store.fetch::<u32, FixtureEngine>(b"non_existent", &mut engine)?,
            None
        );

        assert!(store.fetch::<u64, FixtureEngine>(key, &mut engine).is_err());

        Ok(())
    }

    #[test]
    fn test_unit_store_skips_encode_in_no_cache_mode() -> Result<()> {
        #[derive(Default)]
        struct PanicOnEncode;

        impl CodecEngine<u32> for PanicOnEncode {
            fn encode(&mut self, _: &u32) -> std::result::Result<&[u8], SkipReason> {
                panic!("no-cache store should not encode values");
            }

            fn decode(&mut self, _: &[u8]) -> Result<u32> {
                unreachable!("no-cache fetch always misses")
            }
        }

        let store = ();
        let mut engine = PanicOnEncode;
        store.store::<u32, PanicOnEncode>(b"ignored", &mut engine, &42)?;
        Ok(())
    }

    #[test]
    fn test_unit_store_encoded_is_noop() -> Result<()> {
        let store = ();
        store.store_encoded(b"ignored", b"payload")?;
        Ok(())
    }

    #[cfg(feature = "lz4")]
    #[test]
    fn test_hashmap_store_compressed_value_roundtrip() -> Result<()> {
        use crate::cache::codec::{
            compress::{CompressedCodec, lz4::Lz4},
            fixtures::FixtureEngine,
        };
        type Lz4Engine = CompressedCodec<FixtureEngine, Lz4>;

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

    #[cfg(all(feature = "lz4", feature = "bitcode"))]
    #[test]
    fn test_hashmap_store_skips_oversize_compressed_value() -> Result<()> {
        use crate::cache::codec::compress::{CompressedCodec, fixtures::SizedBytesRaw, lz4::Lz4};
        type Lz4Engine = CompressedCodec<SizedBytesRaw, Lz4>;

        let store = DefaultHashMapStore::default();
        let mut engine = Lz4Engine::default();
        let key = b"oversize";
        let value = 64 * 1024 * 1024 + 1;

        store.store::<usize, Lz4Engine>(key, &mut engine, &value)?;
        assert_eq!(store.fetch::<usize, Lz4Engine>(key, &mut engine)?, None);

        let map = store.0.read().unwrap();
        assert!(!map.contains_key(key.as_slice()));
        Ok(())
    }

    #[cfg(feature = "fjall")]
    #[test]
    fn test_fjall_store() -> Result<()> {
        let tmp = tempfile::tempdir().unwrap();
        let db = fjall::Database::builder(&tmp).open()?;
        let partition = db.keyspace("test", Default::default)?;
        let mut engine = FixtureEngine::default();

        let value = 42u32;
        let key = b"test_sig";
        partition.store::<u32, FixtureEngine>(key, &mut engine, &value)?;
        assert_eq!(
            partition.fetch::<u32, FixtureEngine>(key, &mut engine)?,
            Some(value)
        );

        assert_eq!(
            partition.fetch::<u32, FixtureEngine>(b"non_existent", &mut engine)?,
            None
        );

        assert!(
            partition
                .fetch::<u64, FixtureEngine>(key, &mut engine)
                .is_err()
        );

        Ok(())
    }
}
