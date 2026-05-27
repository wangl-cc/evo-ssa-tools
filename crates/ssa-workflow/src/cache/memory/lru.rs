use std::{collections::hash_map::RandomState, hash::BuildHasher, num::NonZeroUsize, sync::Arc};

use parking_lot::Mutex;

use crate::{
    Result,
    cache::{Cache, CloneShared, memory::ManagedMemoryCache},
};

type LruCache<T, H> = lru::LruCache<Box<[u8]>, T, H>;

/// A bounded in-memory typed LRU cache.
///
/// Stores typed values `T` directly — no codec required. Evicts the least recently used entry
/// when the capacity is exceeded.
///
/// All workers share the same underlying cache: a result stored by one worker is immediately
/// available to the others, and evictions from any worker affect the shared pool.
///
/// This cache is thread-safe, but not single-flight. Concurrent misses for the same key may execute
/// the caller-provided computation more than once.
#[derive(Debug)]
pub struct LruObjectCache<T, H: BuildHasher = RandomState> {
    inner: Arc<Mutex<LruCache<T, H>>>,
}

impl<T> LruObjectCache<T, RandomState> {
    /// Create a typed LRU cache with the given entry capacity.
    pub fn new(capacity: NonZeroUsize) -> Self {
        Self::with_hasher(capacity, RandomState::default())
    }
}

impl<T, H> LruObjectCache<T, H>
where
    H: BuildHasher,
{
    /// Create a typed LRU cache with a custom hasher.
    pub fn with_hasher(capacity: NonZeroUsize, hasher: H) -> Self {
        Self {
            inner: Arc::new(Mutex::new(LruCache::with_hasher(capacity, hasher))),
        }
    }
}

/// [`LruObjectCache`] with the default hasher.
pub type DefaultLruObjectCache<T> = LruObjectCache<T, RandomState>;

/// Managed bounded LRU cache provider.
pub type ManagedLruCache<T> = ManagedMemoryCache<LruObjectCache<T>>;

impl<T> ManagedLruCache<T>
where
    T: Clone + Send + Sync + 'static,
{
    /// Create a managed bounded LRU cache provider.
    pub fn new(capacity: NonZeroUsize) -> Self {
        Self::from_cache(LruObjectCache::new(capacity))
    }
}

impl<T, H> ManagedMemoryCache<LruObjectCache<T, H>>
where
    T: Clone + Send + Sync + 'static,
    H: BuildHasher,
{
    /// Create a managed bounded LRU cache provider with a custom hasher.
    pub fn with_hasher(capacity: NonZeroUsize, hasher: H) -> Self {
        Self::from_cache(LruObjectCache::with_hasher(capacity, hasher))
    }
}

impl<T, H> CloneShared for LruObjectCache<T, H>
where
    T: Send + Sync,
    H: BuildHasher + Send + Sync,
{
    fn clone_shared(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T, H> Cache<T> for LruObjectCache<T, H>
where
    T: Clone + Send + Sync + 'static,
    H: BuildHasher + Send + Sync,
{
    fn fetch(&mut self, key: &[u8]) -> Result<Option<T>> {
        Ok(self.inner.lock().get(key).cloned())
    }

    fn store(&mut self, key: &[u8], value: &T) -> Result<()> {
        self.inner.lock().put(Box::from(key), value.clone());
        Ok(())
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    mod fixtures {
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };

        pub(super) fn entry_capacity() -> std::num::NonZeroUsize {
            std::num::NonZeroUsize::new(2).expect("capacity is non-zero")
        }

        #[derive(Clone, Default)]
        pub(super) struct CallCounter {
            inner: Arc<AtomicUsize>,
        }

        impl CallCounter {
            pub(super) fn increment(&self) {
                self.inner.fetch_add(1, Ordering::SeqCst);
            }

            pub(super) fn get(&self) -> usize {
                self.inner.load(Ordering::SeqCst)
            }
        }
    }

    mod fetch {
        use super::fixtures::{CallCounter, entry_capacity};
        use crate::{
            Result,
            cache::{Cache, memory::DefaultLruObjectCache},
            error::Error,
        };

        #[test]
        fn hit_skips_execute() -> Result<()> {
            let mut cache = DefaultLruObjectCache::new(entry_capacity());
            let calls = CallCounter::default();

            let first = cache.fetch_or_execute(b"same", || {
                calls.increment();
                Ok(String::from("value"))
            })?;
            let second = cache.fetch_or_execute(b"same", || {
                calls.increment();
                Ok(String::from("other"))
            })?;

            assert_eq!(first, "value");
            assert_eq!(second, "value");
            assert_eq!(calls.get(), 1);
            Ok(())
        }

        #[test]
        fn execute_error_does_not_store() -> Result<()> {
            let mut cache = DefaultLruObjectCache::new(entry_capacity());

            let result = cache.fetch_or_execute(b"fallible", || Err(Error::Interrupted));
            assert!(matches!(result, Err(Error::Interrupted)));

            let value = cache.fetch_or_execute(b"fallible", || Ok(12u32))?;
            assert_eq!(value, 12);
            Ok(())
        }
    }

    mod eviction {
        use super::fixtures::{CallCounter, entry_capacity};
        use crate::{
            Result,
            cache::{Cache, memory::DefaultLruObjectCache},
        };

        #[test]
        fn evicts_least_recently_used_entry() -> Result<()> {
            let mut cache = DefaultLruObjectCache::new(entry_capacity());
            let calls = CallCounter::default();

            for key in [
                b"a".as_slice(),
                b"b".as_slice(),
                b"a".as_slice(),
                b"c".as_slice(),
            ] {
                cache.fetch_or_execute(key, || {
                    calls.increment();
                    Ok(String::from_utf8(key.to_vec()).expect("ascii test key"))
                })?;
            }
            let _ = cache.fetch_or_execute(b"b", || {
                calls.increment();
                Ok(String::from("b"))
            })?;

            assert_eq!(calls.get(), 4);
            Ok(())
        }
    }

    mod clone_shared {
        use super::fixtures::{CallCounter, entry_capacity};
        use crate::{
            Result,
            cache::{Cache, CloneShared, memory::DefaultLruObjectCache},
        };

        #[test]
        fn shares_state() -> Result<()> {
            let cache = DefaultLruObjectCache::new(entry_capacity());
            let mut forked = cache.clone_shared();
            let mut cache = cache;
            let calls = CallCounter::default();

            let first = cache.fetch_or_execute(b"shared", || {
                calls.increment();
                Ok(7usize)
            })?;
            let second = forked.fetch_or_execute(b"shared", || {
                calls.increment();
                Ok(9usize)
            })?;

            assert_eq!(first, 7);
            assert_eq!(second, 7);
            assert_eq!(calls.get(), 1);
            Ok(())
        }
    }

    mod formatting {
        use super::fixtures::entry_capacity;
        use crate::{
            Result,
            cache::{Cache, memory::DefaultLruObjectCache},
        };

        #[test]
        fn debug_names_cache_type() {
            let cache = DefaultLruObjectCache::<String>::new(entry_capacity());
            let debug = format!("{cache:?}");
            assert!(debug.contains("LruObjectCache"));
        }

        #[test]
        fn with_hasher_builds_working_cache() -> Result<()> {
            let mut cache = DefaultLruObjectCache::with_hasher(
                entry_capacity(),
                std::collections::hash_map::RandomState::default(),
            );

            let value = cache.fetch_or_execute(b"k", || Ok(42u32))?;
            assert_eq!(value, 42);
            Ok(())
        }

        #[test]
        fn managed_with_hasher_builds_working_cache() -> Result<()> {
            use crate::{
                cache::{memory::ManagedLruCache, provider::CacheProvider},
                identity::{ComputationId, ComputationPath},
            };

            let provider = ManagedLruCache::<u32>::with_hasher(
                entry_capacity(),
                std::collections::hash_map::RandomState::default(),
            );
            let path = ComputationPath::root(ComputationId::new("managed-lru/v1"));
            let mut cache = provider.bind(&path)?;

            let value = cache.fetch_or_execute(b"k", || Ok(42u32))?;
            assert_eq!(value, 42);
            Ok(())
        }
    }
}
