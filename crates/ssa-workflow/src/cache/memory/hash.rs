use std::{
    collections::{HashMap, hash_map::RandomState},
    hash::BuildHasher,
    sync::Arc,
};

use parking_lot::RwLock;

use crate::{
    Result,
    cache::{Cache, CloneShared, memory::ManagedMemoryCache},
};

type RawMap<T, H> = HashMap<Box<[u8]>, T, H>;

/// An unbounded in-memory typed cache backed by a shared `HashMap`.
///
/// Stores typed values `T` directly — no codec required. This is the default in-process cache
/// for tests, benchmarks, and workflows that only need cache reuse within one process.
///
/// All workers share the same underlying map: a result stored by one worker is immediately
/// available to the others.
///
/// This cache is thread-safe, but not single-flight. Concurrent misses for the same key may execute
/// the caller-provided computation more than once.
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

/// Managed unbounded hash cache provider.
pub type ManagedHashCache<T> = ManagedMemoryCache<HashObjectCache<T>>;

impl<T> ManagedHashCache<T>
where
    T: Clone + Send + Sync + 'static,
{
    /// Create a managed unbounded hash cache provider.
    pub fn new() -> Self {
        Self::from_cache(HashObjectCache::default())
    }
}

impl<T, H> ManagedMemoryCache<HashObjectCache<T, H>>
where
    T: Clone + Send + Sync + 'static,
    H: BuildHasher,
{
    /// Create a managed unbounded hash cache provider with a custom hasher.
    pub fn with_hasher(hasher: H) -> Self {
        Self::from_cache(HashObjectCache::with_hasher(hasher))
    }
}

impl<T> Default for ManagedHashCache<T>
where
    T: Clone + Send + Sync + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T, H> CloneShared for HashObjectCache<T, H>
where
    T: Send + Sync,
    H: Send + Sync,
{
    fn clone_shared(&self) -> Self {
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
    fn fetch(&mut self, key: &[u8]) -> Result<Option<T>> {
        Ok(self.inner.read().get(key).cloned())
    }

    fn store(&mut self, key: &[u8], value: &T) -> Result<()> {
        self.inner.write().insert(Box::from(key), value.clone());
        Ok(())
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use std::sync::{
        Arc, Barrier,
        atomic::{AtomicUsize, Ordering},
    };

    use super::*;
    use crate::error::Error;

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
    fn test_hash_object_cache_execute_error_does_not_store() -> Result<()> {
        let mut cache = DefaultHashObjectCache::<u32>::default();

        let result = cache.fetch_or_execute(b"fallible", || Err(Error::Interrupted));
        assert!(matches!(result, Err(Error::Interrupted)));

        let value = cache.fetch_or_execute(b"fallible", || Ok(12u32))?;
        assert_eq!(value, 12);
        Ok(())
    }

    #[test]
    fn test_hash_object_cache_is_thread_safe_but_not_single_flight() -> Result<()> {
        let cache = DefaultHashObjectCache::<usize>::default();
        let mut first_cache = cache.clone_shared();
        let mut second_cache = cache.clone_shared();
        let started = Arc::new(Barrier::new(2));
        let calls = Arc::new(AtomicUsize::new(0));

        let first = std::thread::spawn({
            let started = Arc::clone(&started);
            let calls = Arc::clone(&calls);
            move || {
                first_cache.fetch_or_execute(b"same", || {
                    calls.fetch_add(1, Ordering::SeqCst);
                    started.wait();
                    Ok(1usize)
                })
            }
        });
        let second = std::thread::spawn({
            let started = Arc::clone(&started);
            let calls = Arc::clone(&calls);
            move || {
                second_cache.fetch_or_execute(b"same", || {
                    calls.fetch_add(1, Ordering::SeqCst);
                    started.wait();
                    Ok(2usize)
                })
            }
        });

        let first = first.join().expect("worker should not panic")?;
        let second = second.join().expect("worker should not panic")?;

        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_ne!(first, second);
        Ok(())
    }

    #[test]
    fn test_hash_object_cache_clone_shared_shares_state() -> Result<()> {
        let cache = DefaultHashObjectCache::default();
        let mut forked = cache.clone_shared();
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

    #[test]
    fn test_managed_hash_cache_with_hasher() -> Result<()> {
        use crate::{
            cache::provider::CacheProvider,
            identity::{ComputationId, ComputationPath},
        };

        let provider = ManagedHashCache::<u32>::with_hasher(
            std::collections::hash_map::RandomState::default(),
        );
        let path = ComputationPath::root(ComputationId::new("managed-hash/v1"));
        let mut cache = provider.bind(&path)?;

        let value = cache.fetch_or_execute(b"k", || Ok(42u32))?;
        assert_eq!(value, 42);
        Ok(())
    }

    #[test]
    fn test_hash_object_cache_debug_format() {
        let cache = DefaultHashObjectCache::<String>::default();
        let debug = format!("{cache:?}");
        assert!(debug.contains("HashObjectCache"));
    }
}
