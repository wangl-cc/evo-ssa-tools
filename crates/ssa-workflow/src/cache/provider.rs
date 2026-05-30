use super::{
    Cache, CanonicalEncode, CloneShared, EncodedCache,
    codec::{CloneFresh, CodecEngine},
    storage::{EncodedStorage, StorageNamespace},
};
use crate::{Result, identity::ComputationPath};

/// Provider that binds a semantic computation path to an execution cache.
pub trait CacheProvider<T> {
    /// Bound cache used by the execution path.
    type Cache: Cache<T> + CloneShared;

    /// Bind the provider to a computation path and input schema, consuming it.
    fn bind<I: CanonicalEncode>(self, path: &ComputationPath) -> Result<Self::Cache>;
}

impl<T> CacheProvider<T> for () {
    type Cache = ();

    fn bind<I: CanonicalEncode>(self, _path: &ComputationPath) -> Result<Self::Cache> {
        Ok(())
    }
}

/// Storage provider root that opens raw namespaced storage.
pub trait StorageProvider: Clone + Send + Sync + 'static {
    /// Bound storage type returned for one namespace.
    type Storage: EncodedStorage + CloneShared;

    /// Open or create a raw namespace storage.
    fn open_storage(&self, namespace: &StorageNamespace) -> Result<Self::Storage>;
}

/// Convenience methods for storage providers.
pub trait StorageProviderExt: StorageProvider + Sized {
    /// Attach a codec to this storage provider, producing a persistent cache provider.
    fn with_codec<CE>(self, codec: CE) -> PersistentCacheProvider<Self, CE> {
        PersistentCacheProvider::new(self, codec)
    }
}

impl<SP: StorageProvider> StorageProviderExt for SP {}

/// Cache provider composed from a storage provider and a codec.
pub struct PersistentCacheProvider<SP, CE> {
    storage_provider: SP,
    codec: CE,
}

impl<SP, CE> PersistentCacheProvider<SP, CE> {
    /// Create a cache provider from a storage provider and codec.
    pub fn new(storage_provider: SP, codec: CE) -> Self {
        Self {
            storage_provider,
            codec,
        }
    }
}

impl<SP: Clone, CE: CloneFresh> Clone for PersistentCacheProvider<SP, CE> {
    fn clone(&self) -> Self {
        Self {
            storage_provider: self.storage_provider.clone(),
            codec: self.codec.clone_fresh(),
        }
    }
}

impl<T, SP, CE> CacheProvider<T> for PersistentCacheProvider<SP, CE>
where
    SP: StorageProvider,
    CE: CodecEngine<T> + CloneFresh + Send + Sync + 'static,
{
    type Cache = EncodedCache<SP::Storage, CE>;

    fn bind<I: CanonicalEncode>(self, path: &ComputationPath) -> Result<Self::Cache> {
        let namespace = StorageNamespace::new::<I>(path, CE::VALUE_FORMAT);
        let storage = self.storage_provider.open_storage(&namespace)?;
        Ok(EncodedCache::new(storage, self.codec.clone_fresh()))
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::{
        Error,
        cache::{
            codec::fixtures::FixtureEngine,
            storage::{EncodedStorage, StorageResult},
        },
        identity::ComputationPath,
    };

    #[derive(Clone, Copy)]
    struct FailingStorageProvider;

    struct NeverStorage;

    impl CloneShared for NeverStorage {
        fn clone_shared(&self) -> Self {
            Self
        }
    }

    impl EncodedStorage for NeverStorage {
        type Encoded<'a> = &'a [u8];

        fn fetch_encoded(&self, _key: &[u8]) -> StorageResult<Option<Self::Encoded<'_>>> {
            Ok(None)
        }

        fn store_encoded(&self, _key: &[u8], _encoded: &[u8]) -> StorageResult<()> {
            Ok(())
        }
    }

    impl StorageProvider for FailingStorageProvider {
        type Storage = NeverStorage;

        fn open_storage(&self, _namespace: &StorageNamespace) -> Result<Self::Storage> {
            Err(Error::Compute(
                std::io::Error::other("open storage failed").into(),
            ))
        }
    }

    #[test]
    fn persistent_cache_provider_bind_propagates_storage_open_error() {
        let provider =
            PersistentCacheProvider::new(FailingStorageProvider, FixtureEngine::default());
        let path = ComputationPath::root_from_str("provider-bind-failure-v1");

        let result = <_ as CacheProvider<u32>>::bind::<u16>(provider, &path);

        assert!(matches!(result, Err(Error::Compute(_))));
    }
}
