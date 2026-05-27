#[cfg(feature = "lru")]
use core::num::NonZeroUsize;
use std::{marker::PhantomData, sync::Arc};

use parking_lot::Mutex;

use super::HashObjectCache;
#[cfg(feature = "lru")]
use super::LruObjectCache;
use crate::{
    Result,
    cache::{Cache, CloneShared, provider::CacheProvider},
    identity::ComputationPath,
};

/// Strategy that creates one concrete in-memory cache space.
pub trait MemoryCacheStorage<T>: Clone + Send + Sync + 'static {
    /// Bound cache type created by this strategy.
    type Cache: Cache<T> + CloneShared;

    /// Create one empty cache space.
    fn new_cache(&self) -> Self::Cache;
}

/// Unbounded hash-map backed memory cache strategy.
#[derive(Clone, Copy, Debug, Default)]
pub struct HashMemory;

impl<T> MemoryCacheStorage<T> for HashMemory
where
    T: Clone + Send + Sync + 'static,
{
    type Cache = HashObjectCache<T>;

    fn new_cache(&self) -> Self::Cache {
        HashObjectCache::default()
    }
}

/// Bounded LRU memory cache strategy.
#[cfg(feature = "lru")]
#[derive(Clone, Copy, Debug)]
pub struct LruMemory {
    capacity: NonZeroUsize,
}

#[cfg(feature = "lru")]
impl LruMemory {
    /// Create an LRU strategy with the given entry capacity.
    pub const fn new(capacity: NonZeroUsize) -> Self {
        Self { capacity }
    }
}

#[cfg(feature = "lru")]
impl<T> MemoryCacheStorage<T> for LruMemory
where
    T: Clone + Send + Sync + 'static,
{
    type Cache = LruObjectCache<T>;

    fn new_cache(&self) -> Self::Cache {
        LruObjectCache::new(self.capacity)
    }
}

struct BoundMemorySpace<C> {
    cache: C,
}

/// Single-space typed managed memory cache provider.
pub struct ManagedMemoryCache<T, S = HashMemory>
where
    S: MemoryCacheStorage<T>,
{
    strategy: S,
    state: Arc<Mutex<Option<BoundMemorySpace<S::Cache>>>>,
    _marker: PhantomData<fn() -> T>,
}

impl<T, S> ManagedMemoryCache<T, S>
where
    S: MemoryCacheStorage<T>,
{
    /// Create a managed memory cache from a storage strategy.
    pub fn with_strategy(strategy: S) -> Self {
        Self {
            strategy,
            state: Arc::new(Mutex::new(None)),
            _marker: PhantomData,
        }
    }
}

impl<T> ManagedMemoryCache<T, HashMemory>
where
    T: Clone + Send + Sync + 'static,
{
    /// Create a managed unbounded hash cache.
    pub fn hash() -> Self {
        Self::with_strategy(HashMemory)
    }
}

#[cfg(feature = "lru")]
impl<T> ManagedMemoryCache<T, LruMemory>
where
    T: Clone + Send + Sync + 'static,
{
    /// Create a managed bounded LRU cache.
    pub fn lru(capacity: NonZeroUsize) -> Self {
        Self::with_strategy(LruMemory::new(capacity))
    }
}

impl<T> Default for ManagedMemoryCache<T, HashMemory>
where
    T: Clone + Send + Sync + 'static,
{
    fn default() -> Self {
        Self::hash()
    }
}

impl<T, S> Clone for ManagedMemoryCache<T, S>
where
    S: MemoryCacheStorage<T>,
{
    fn clone(&self) -> Self {
        Self {
            strategy: self.strategy.clone(),
            state: self.state.clone(),
            _marker: PhantomData,
        }
    }
}

impl<T, S> CacheProvider<T> for ManagedMemoryCache<T, S>
where
    S: MemoryCacheStorage<T>,
{
    type Cache = S::Cache;

    fn bind(self, _path: &ComputationPath) -> Result<Self::Cache> {
        let mut state = self.state.lock();
        if let Some(bound) = &*state {
            return Ok(bound.cache.clone_shared());
        }
        let cache = self.strategy.new_cache();
        let bound_cache = cache.clone_shared();
        *state = Some(BoundMemorySpace { cache });
        Ok(bound_cache)
    }
}

/// Managed unbounded hash cache provider.
pub type ManagedHashCache<T> = ManagedMemoryCache<T, HashMemory>;

/// Managed bounded LRU cache provider.
#[cfg(feature = "lru")]
pub type ManagedLruCache<T> = ManagedMemoryCache<T, LruMemory>;

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::identity::{ComputationId, ComputationPath};

    #[test]
    fn same_path_reuses_hash_cache() -> Result<()> {
        let provider = ManagedHashCache::<u32>::default();
        let path = ComputationPath::root(ComputationId::new("answer/v1"));
        let mut first = provider.clone().bind(&path)?;
        let mut second = provider.bind(&path)?;

        let value = first.fetch_or_execute(b"k", || Ok(7))?;
        let reused = second.fetch_or_execute(b"k", || Ok(9))?;

        assert_eq!(value, 7);
        assert_eq!(reused, 7);
        Ok(())
    }
}
