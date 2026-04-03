//! Typed cache adapter backed by a raw [`CacheStore`](crate::cache::storage::CacheStore).

use super::{Cache, Fork, codec::CodecEngine, storage::CacheStore};
use crate::error::Result;

/// A cache backed by a raw [`CacheStore`] and a [`CodecEngine`].
///
/// `EncodedCache` owns both a storage backend and a codec engine. When passed to `execute_many`,
/// each worker gets its own instance: the store is forked into a shared handle (all workers read
/// and write the same underlying data), and the codec is forked into an independent engine with
/// its own encode buffer.
#[derive(Debug)]
pub struct EncodedCache<S, CE> {
    store: S,
    engine: CE,
}

impl<S, CE> EncodedCache<S, CE> {
    /// Create an encoded cache from a raw store and a codec engine.
    pub fn new(store: S, engine: CE) -> Self {
        Self { store, engine }
    }
}

impl<S: Fork, CE: Fork> Fork for EncodedCache<S, CE> {
    fn fork(&self) -> Self {
        Self {
            store: self.store.fork(),
            engine: self.engine.fork(),
        }
    }
}

impl<S, CE, T> Cache<T> for EncodedCache<S, CE>
where
    S: CacheStore,
    CE: CodecEngine<T>,
{
    fn fetch_or_execute<F>(&mut self, key: &[u8], execute: F) -> Result<T>
    where
        F: FnOnce() -> Result<T>,
    {
        if let Some(cached) = self.store.fetch::<T, CE>(key, &mut self.engine)? {
            return Ok(cached);
        }
        let output = execute()?;
        self.store.store::<T, CE>(key, &mut self.engine, &output)?;
        Ok(output)
    }
}
