//! Typed cache adapter backed by a raw [`CacheStore`](crate::cache::storage::CacheStore).

use super::{Cache, CloneFresh, CloneShared, codec::CodecEngine, storage::CacheStore};
use crate::error::Result;

/// A cache backed by a raw [`CacheStore`] and a [`CodecEngine`].
///
/// `EncodedCache` owns both a storage backend and a codec engine. When passed to `execute_many`,
/// each worker gets its own cache handle: the store is cloned as a shared handle, while the codec
/// is cloned fresh with independent worker-local state.
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

impl<S: CloneShared, CE: CloneFresh> CloneShared for EncodedCache<S, CE> {
    fn clone_shared(&self) -> Self {
        Self {
            store: self.store.clone_shared(),
            engine: self.engine.clone_fresh(),
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
