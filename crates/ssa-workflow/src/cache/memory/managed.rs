#[cfg(feature = "lru")]
use core::num::NonZeroUsize;

use super::HashObjectCache;
#[cfg(feature = "lru")]
use super::LruObjectCache;
use crate::{
    Result,
    cache::{Cache, CloneShared, provider::CacheProvider},
    identity::ComputationPath,
};

/// One-shot managed memory cache provider.
///
/// The provider owns one private in-memory cache space. Binding consumes the provider and returns
/// the owned cache; worker and cloned-task sharing happens through the bound cache's
/// [`CloneShared`] implementation.
pub struct ManagedMemoryCache<C> {
    cache: C,
}

impl<C> ManagedMemoryCache<C> {
    /// Create a managed memory provider from an already constructed cache.
    pub fn new(cache: C) -> Self {
        Self { cache }
    }
}

impl<T> ManagedHashCache<T>
where
    T: Clone + Send + Sync + 'static,
{
    /// Create a managed unbounded hash cache provider.
    pub fn hash() -> Self {
        Self::new(HashObjectCache::default())
    }
}

impl<T> Default for ManagedHashCache<T>
where
    T: Clone + Send + Sync + 'static,
{
    fn default() -> Self {
        Self::hash()
    }
}

#[cfg(feature = "lru")]
impl<T> ManagedLruCache<T>
where
    T: Clone + Send + Sync + 'static,
{
    /// Create a managed bounded LRU cache provider.
    pub fn lru(capacity: NonZeroUsize) -> Self {
        Self::new(LruObjectCache::new(capacity))
    }
}

impl<T, C> CacheProvider<T> for ManagedMemoryCache<C>
where
    C: Cache<T> + CloneShared,
{
    type Cache = C;

    fn bind(self, _path: &ComputationPath) -> Result<Self::Cache> {
        Ok(self.cache)
    }
}

/// Managed unbounded hash cache provider.
pub type ManagedHashCache<T> = ManagedMemoryCache<HashObjectCache<T>>;

/// Managed bounded LRU cache provider.
#[cfg(feature = "lru")]
pub type ManagedLruCache<T> = ManagedMemoryCache<LruObjectCache<T>>;

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::identity::{ComputationId, ComputationPath};

    #[test]
    fn bind_returns_owned_hash_cache() -> Result<()> {
        let provider = ManagedHashCache::<u32>::default();
        let path = ComputationPath::root(ComputationId::new("answer/v1"));
        let mut cache = provider.bind(&path)?;

        let value = cache.fetch_or_execute(b"k", || Ok(7))?;
        let reused = cache.fetch_or_execute(b"k", || Ok(9))?;

        assert_eq!(value, 7);
        assert_eq!(reused, 7);
        Ok(())
    }
}
