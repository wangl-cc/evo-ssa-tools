use super::{
    Cache, CloneShared, EncodedCache,
    codec::{CloneFresh, CodecEngine},
    storage::{CacheStore, StorageNamespace},
};
use crate::{Result, identity::ComputationPath};

/// Provider that binds a semantic computation path to an execution cache.
pub trait CacheProvider<T>: Clone {
    /// Bound cache used by the execution path.
    type Cache: Cache<T> + CloneShared;

    /// Bind the provider to a computation path.
    fn bind(&self, path: &ComputationPath) -> Result<Self::Cache>;
}

impl<T> CacheProvider<T> for () {
    type Cache = ();

    fn bind(&self, _path: &ComputationPath) -> Result<Self::Cache> {
        Ok(())
    }
}

/// Persistent backend root that can open a store for a storage namespace.
pub trait PersistentBackend: Clone + Send + Sync + 'static {
    /// Bound store type returned for one namespace.
    type Store: CacheStore + CloneShared;

    /// Open or create the backend namespace.
    fn open_namespace(&self, namespace: &StorageNamespace) -> Result<Self::Store>;
}

/// Multi-space persistent cache provider.
pub struct ManagedPersistentCache<B, CE> {
    backend: B,
    codec: CE,
}

impl<B, CE> ManagedPersistentCache<B, CE> {
    /// Create a managed persistent cache provider from a backend root and codec.
    pub fn new(backend: B, codec: CE) -> Self {
        Self { backend, codec }
    }
}

impl<B: Clone, CE: CloneFresh> Clone for ManagedPersistentCache<B, CE> {
    fn clone(&self) -> Self {
        Self {
            backend: self.backend.clone(),
            codec: self.codec.clone_fresh(),
        }
    }
}

impl<T, B, CE> CacheProvider<T> for ManagedPersistentCache<B, CE>
where
    B: PersistentBackend,
    CE: CodecEngine<T> + CloneFresh + Send + Sync + 'static,
{
    type Cache = EncodedCache<B::Store, CE>;

    fn bind(&self, path: &ComputationPath) -> Result<Self::Cache> {
        let namespace = StorageNamespace::new(path, CE::VALUE_FORMAT);
        let store = self.backend.open_namespace(&namespace)?;
        Ok(EncodedCache::new(store, self.codec.clone_fresh()))
    }
}
