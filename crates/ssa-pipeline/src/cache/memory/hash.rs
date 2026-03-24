use std::{
    collections::{HashMap, hash_map::RandomState},
    hash::BuildHasher,
    sync::Arc,
};

use parking_lot::RwLock;

use crate::{
    Result,
    cache::{Cache, Fork},
};

type RawMap<T, H> = HashMap<Box<[u8]>, T, H>;

/// An unbounded in-memory typed cache backed by a shared `HashMap`.
///
/// Stores typed values `T` directly — no codec required. This is the default in-process cache
/// for tests, benchmarks, and pipelines that only need cache reuse within one process.
///
/// All workers share the same underlying map: a result stored by one worker is immediately
/// available to the others.
#[derive(Debug)]
pub struct HashObjectCache<T, H = RandomState> {
    inner: Arc<RwLock<RawMap<T, H>>>,
}

impl<T, H> Default for HashObjectCache<T, H>
where
    H: BuildHasher + Default,
{
    fn default() -> Self {
        Self {
            inner: Arc::new(RwLock::new(RawMap::default())),
        }
    }
}

impl<T, H> HashObjectCache<T, H>
where
    H: BuildHasher,
{
    /// Create a typed cache with a custom hasher.
    pub fn with_hasher(hasher: H) -> Self {
        Self {
            inner: Arc::new(RwLock::new(RawMap::with_hasher(hasher))),
        }
    }
}

/// [`HashObjectCache`] with the default hasher.
pub type DefaultHashObjectCache<T> = HashObjectCache<T, RandomState>;

impl<T, H> Fork for HashObjectCache<T, H>
where
    T: Send + Sync,
    H: Send + Sync,
{
    fn fork(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T, H> Cache<T> for HashObjectCache<T, H>
where
    T: Clone + Send + Sync + 'static,
    H: BuildHasher + Send + Sync,
{
    fn fetch_or_execute<F>(&mut self, key: &[u8], execute: F) -> Result<T>
    where
        F: FnOnce() -> Result<T>,
    {
        if let Some(cached) = self.inner.read().get(key).cloned() {
            return Ok(cached);
        }
        let output = execute()?;
        self.inner.write().insert(Box::from(key), output.clone());
        Ok(output)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use super::*;

    #[test]
    fn test_hash_object_cache_hits_skip_execute() -> Result<()> {
        let mut cache = DefaultHashObjectCache::<String>::default();
        let calls = Arc::new(AtomicUsize::new(0));

        let first = cache.fetch_or_execute(b"same", || {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(String::from("value"))
        })?;
        let second = cache.fetch_or_execute(b"same", || {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(String::from("other"))
        })?;

        assert_eq!(first, "value");
        assert_eq!(second, "value");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[test]
    fn test_hash_object_cache_distinct_keys_are_independent() -> Result<()> {
        let mut cache = DefaultHashObjectCache::<u32>::default();

        let a = cache.fetch_or_execute(b"a", || Ok(1u32))?;
        let b = cache.fetch_or_execute(b"b", || Ok(2u32))?;

        assert_eq!(a, 1);
        assert_eq!(b, 2);
        Ok(())
    }

    #[test]
    fn test_hash_object_cache_debug_format() {
        let cache = DefaultHashObjectCache::<String>::default();
        let debug = format!("{cache:?}");
        assert!(debug.contains("HashObjectCache"));
    }

    #[test]
    fn test_hash_object_cache_fork_shares_state() -> Result<()> {
        let cache = DefaultHashObjectCache::default();
        let mut forked = cache.fork();
        let mut cache = cache;
        let calls = Arc::new(AtomicUsize::new(0));

        let first = cache.fetch_or_execute(b"shared", || {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(7usize)
        })?;
        let second = forked.fetch_or_execute(b"shared", || {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(9usize)
        })?;

        assert_eq!(first, 7);
        assert_eq!(second, 7);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[test]
    fn test_hash_object_cache_with_hasher() -> Result<()> {
        let mut cache =
            HashObjectCache::with_hasher(std::collections::hash_map::RandomState::default());
        let v = cache.fetch_or_execute(b"k", || Ok(42u32))?;
        assert_eq!(v, 42);
        Ok(())
    }
}
