use std::sync::{Arc, RwLock};

use super::{CacheStore, Result, WorkerForkStore, private};

type RawHashMap<H> = std::collections::HashMap<Vec<u8>, Vec<u8>, H>;
type HashMapShard<H> = Arc<RwLock<RawHashMap<H>>>;

/// A simple in-memory cache store backed by a `HashMap`.
///
/// Clone is intentionally not exposed. Worker sharing is handled internally.
#[derive(Debug)]
pub struct HashMapStore<H> {
    pub(crate) inner: HashMapShard<H>,
}

impl<H> Default for HashMapStore<H>
where
    H: std::hash::BuildHasher + Default,
{
    fn default() -> Self {
        Self {
            inner: Arc::new(RwLock::new(RawHashMap::<H>::default())),
        }
    }
}

/// Type alias for a [`HashMapStore`] using the default [`HashMap`] hasher.
pub type DefaultHashMapStore = HashMapStore<std::collections::hash_map::RandomState>;

impl<H> private::Sealed for HashMapStore<H> {}

impl<H> WorkerForkStore for HashMapStore<H>
where
    H: std::hash::BuildHasher + Send + Sync,
{
    fn fork_store(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<H> CacheStore for HashMapStore<H>
where
    H: std::hash::BuildHasher + Send + Sync,
{
    fn fetch_encoded_with<T, E, F>(&self, key: &[u8], f: F) -> std::result::Result<T, E>
    where
        F: FnOnce(Option<&[u8]>) -> std::result::Result<T, E>,
        E: From<super::Error>,
    {
        let map = self
            .inner
            .read()
            .expect("cache read lock poisoned: HashMapStore should not panic while holding lock");
        let encoded = map.get(key).cloned();
        drop(map);
        f(encoded.as_deref())
    }

    fn store_encoded(&self, key: &[u8], encoded: &[u8]) -> Result<()> {
        let mut map = self
            .inner
            .write()
            .expect("cache write lock poisoned: HashMapStore should not panic while holding lock");
        map.insert(key.to_owned(), encoded.to_vec());
        Ok(())
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::{cache::codec::fixtures::FixtureEngine, error::Result};

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

    #[cfg(feature = "lz4")]
    #[test]
    fn test_hashmap_store_compressed_value_roundtrip() -> Result<()> {
        use crate::cache::codec::{
            compress::{CompressedCodec, algorithm::Lz4},
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

        let map = store.inner.read().unwrap();
        let encoded = map.get(key.as_slice()).expect("value should be stored");
        assert!(!encoded.is_empty());
        assert_eq!(encoded[0] & 0b1111_0000, 0b0001_0000);
        assert_eq!(encoded[0] & 0b0000_1111, 0b0001);
        Ok(())
    }

    #[cfg(all(feature = "lz4", feature = "bitcode"))]
    #[test]
    fn test_hashmap_store_skips_oversize_compressed_value() -> Result<()> {
        use crate::cache::codec::compress::{
            CompressedCodec, algorithm::Lz4, fixtures::SizedBytesEngine,
        };

        type Lz4Engine = CompressedCodec<SizedBytesEngine, Lz4>;

        let store = DefaultHashMapStore::default();
        let mut engine = Lz4Engine::default().with_max_encode_len(64 * 1024 * 1024);
        let key = b"oversize";
        let value = 64 * 1024 * 1024 + 1;

        store.store::<usize, Lz4Engine>(key, &mut engine, &value)?;
        assert_eq!(store.fetch::<usize, Lz4Engine>(key, &mut engine)?, None);
        Ok(())
    }
}
