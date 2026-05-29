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
///
/// The computation path passed to `bind` is intentionally ignored. Managed memory providers are
/// one-shot and path-agnostic; isolation comes from provider ownership rather than path-derived
/// namespacing.
///
/// This wrapper is intentionally not directly constructible by downstream callers. Use concrete
/// providers such as `ManagedHashCache` and `ManagedLruCache`, or implement [`CacheProvider`] for
/// custom provider types that need different keyspace semantics.
pub struct ManagedMemoryCache<C> {
    cache: C,
}

impl<C> ManagedMemoryCache<C> {
    pub(super) fn from_cache(cache: C) -> Self {
        Self { cache }
    }
}

impl<T, C> CacheProvider<T> for ManagedMemoryCache<C>
where
    C: Cache<T> + CloneShared,
{
    type Cache = C;

    fn bind(self, _: &ComputationPath) -> Result<Self::Cache> {
        Ok(self.cache)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::{cache::memory::HashObjectCache, identity::ComputationPath};

    #[test]
    fn bind_returns_owned_hash_cache() -> Result<()> {
        let provider = ManagedMemoryCache::from_cache(HashObjectCache::<u32>::default());
        let path = ComputationPath::root_from_str("answer-v1");
        let mut cache = provider.bind(&path)?;

        let value = cache.fetch_or_execute(b"k", || Ok(7))?;
        let reused = cache.fetch_or_execute(b"k", || Ok(9))?;

        assert_eq!(value, 7);
        assert_eq!(reused, 7);
        Ok(())
    }
}
