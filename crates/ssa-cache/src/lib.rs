use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use rayon::prelude::*;

mod error;
pub use error::{Error, Result};

mod cache;
pub use cache::{CacheStore, Cacheable, CodecBuffer, Encodeable, HashMapStore};

/// Core trait for all compute operations
pub trait Compute<C> {
    type Input;
    type Output: Cacheable;

    /// Execute the computation or fetch from cache
    ///
    /// If cache hit, return the cached value. Otherwise, execute the computation and store the
    /// result in the cache.
    fn execute(
        &mut self,
        input: Self::Input,
        cache: &C,
        buffer: &mut <Self::Output as Encodeable>::Buffer,
    ) -> Result<Self::Output>;

    /// Execute many computations in parallel
    fn execute_many(
        &self,
        cache: &C,
        interrupt: Arc<AtomicBool>,
        inputs: impl ParallelIterator<Item = Self::Input>,
    ) -> Result<impl ParallelIterator<Item = Result<Self::Output>>>
    where
        C: Sync,
        Self: Clone + Sync,
        Self::Output: Send,
    {
        Ok(inputs.map_init(
            || (<Self::Output as Encodeable>::Buffer::init(), self.clone()),
            move |(buffer, c), input| {
                if interrupt.load(Ordering::Relaxed) {
                    return Err(Error::Interrupted);
                };

                c.execute(input, cache, buffer)
            },
        ))
    }
}

pub fn fetch_or_execute<C, O, F>(
    sig: &[u8],
    cache: &C,
    buffer: &mut O::Buffer,
    execute: F,
) -> Result<O>
where
    C: CacheStore,
    O: Cacheable,
    F: FnOnce() -> O,
{
    if let Some(cached) = cache.fetch(sig, buffer)? {
        Ok(cached)
    } else {
        let output = execute();
        cache.store(sig, buffer, &output)?;
        Ok(output)
    }
}

mod single;
pub use single::{PureCompute, StochasiticCompute};

mod multi;
pub use multi::ExpAnalysis;
