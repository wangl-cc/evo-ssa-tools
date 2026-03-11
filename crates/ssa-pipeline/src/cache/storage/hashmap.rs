//! In-memory storage backend backed by a shared [`std::collections::HashMap`].

use std::sync::Arc;

use parking_lot::{MappedRwLockReadGuard, RwLock, RwLockReadGuard};

use super::{CacheStore, StorageResult, WorkerForkStore, private};

type RawHashMap<H> = std::collections::HashMap<Vec<u8>, Vec<u8>, H>;
type HashMapShard<H> = Arc<RwLock<RawHashMap<H>>>;

#[doc(hidden)]
pub struct HashMapEncoded<'a>(MappedRwLockReadGuard<'a, [u8]>);

impl AsRef<[u8]> for HashMapEncoded<'_> {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

/// A simple in-memory cache store backed by a `HashMap`.
///
/// This backend is process-local and does not persist data across runs. It is a good default for
/// tests, benchmarks, and pipelines that only need cache reuse within one process.
///
/// Clone is intentionally not exposed. Parallel worker sharing is handled through
/// [`WorkerForkStore`], which keeps all worker-local clones pointed at the same underlying map.
#[derive(Debug)]
pub struct HashMapStore<H> {
    pub(crate) inner: HashMapShard<H>,
}

impl<H> Default for HashMapStore<H>
where
    H: std::hash::BuildHasher + Default,
{
    /// Create an empty in-memory store using the default hasher instance.
    fn default() -> Self {
        Self {
            inner: Arc::new(RwLock::new(RawHashMap::<H>::default())),
        }
    }
}

/// Type alias for a [`HashMapStore`] using [`std::collections::hash_map::RandomState`].
///
/// This is the easiest in-memory backend to reach for when you do not need a custom hashing
/// strategy.
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
    type Encoded<'a>
        = HashMapEncoded<'a>
    where
        Self: 'a;

    fn fetch_encoded(&self, key: &[u8]) -> StorageResult<Option<Self::Encoded<'_>>> {
        let map = self.inner.read();

        match RwLockReadGuard::try_map(map, |map| map.get(key).map(Vec::as_slice)) {
            Ok(encoded) => Ok(Some(HashMapEncoded(encoded))),
            Err(_) => Ok(None),
        }
    }

    fn store_encoded(&self, key: &[u8], encoded: &[u8]) -> StorageResult<()> {
        let mut map = self.inner.write();
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

        let map = store.inner.read();
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
        let mut engine = Lz4Engine::default().with_max_len(64 * 1024 * 1024);
        let key = b"oversize";
        let value = 64 * 1024 * 1024 + 1;

        store.store::<usize, Lz4Engine>(key, &mut engine, &value)?;
        assert_eq!(store.fetch::<usize, Lz4Engine>(key, &mut engine)?, None);
        Ok(())
    }
}
