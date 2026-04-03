use std::{collections::hash_map::RandomState, hash::BuildHasher, num::NonZeroUsize, sync::Arc};

use parking_lot::Mutex;

use crate::{
    Result,
    cache::{Cache, Fork},
};

type LruCache<T, H> = lru::LruCache<Box<[u8]>, T, H>;

/// A bounded in-memory typed LRU cache.
///
/// Stores typed values `T` directly — no codec required. Evicts the least recently used entry
/// when the capacity is exceeded.
///
/// All workers share the same underlying cache: a result stored by one worker is immediately
/// available to the others, and evictions from any worker affect the shared pool.
pub struct LruObjectCache<T, H = RandomState> {
    inner: Arc<Mutex<LruCache<T, H>>>,
}

impl<T, H> core::fmt::Debug for LruObjectCache<T, H> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("LruObjectCache").finish_non_exhaustive()
    }
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

impl<T, H> Fork for LruObjectCache<T, H>
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

impl<T, H> Cache<T> for LruObjectCache<T, H>
where
    T: Clone + Send + Sync + 'static,
    H: BuildHasher + Send + Sync,
{
    fn fetch_or_execute<F>(&mut self, key: &[u8], execute: F) -> Result<T>
    where
        F: FnOnce() -> Result<T>,
    {
        if let Some(cached) = self.inner.lock().get(key).cloned() {
            return Ok(cached);
        }
        let output = execute()?;
        self.inner.lock().put(Box::from(key), output.clone());
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
    fn test_lru_object_cache_hits_skip_execute() -> Result<()> {
        let mut cache =
            DefaultLruObjectCache::new(NonZeroUsize::new(2).expect("capacity is non-zero"));
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
    fn test_lru_object_cache_evicts_least_recently_used_entry() -> Result<()> {
        let mut cache =
            DefaultLruObjectCache::new(NonZeroUsize::new(2).expect("capacity is non-zero"));
        let calls = Arc::new(AtomicUsize::new(0));

        for key in [
            b"a".as_slice(),
            b"b".as_slice(),
            b"a".as_slice(),
            b"c".as_slice(),
        ] {
            cache.fetch_or_execute(key, || {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(String::from_utf8(key.to_vec()).expect("ascii test key"))
            })?;
        }
        let _ = cache.fetch_or_execute(b"b", || {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(String::from("b"))
        })?;

        assert_eq!(calls.load(Ordering::SeqCst), 4);
        Ok(())
    }

    #[test]
    fn test_lru_object_cache_debug_format() {
        let cache = DefaultLruObjectCache::<String>::new(NonZeroUsize::new(4).unwrap());
        let debug = format!("{cache:?}");
        assert!(debug.contains("LruObjectCache"));
    }

    #[test]
    fn test_lru_object_cache_fork_shares_state() -> Result<()> {
        use crate::cache::Fork;
        let cache = DefaultLruObjectCache::new(NonZeroUsize::new(2).expect("capacity is non-zero"));
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
}
