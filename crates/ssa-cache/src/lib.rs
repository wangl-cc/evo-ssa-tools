use rayon::prelude::*;

mod error;
pub use error::{Error, Result};

mod cache;
pub use cache::{CacheStore, Cacheable, CodecBuffer, Encodeable, HashMapStore};

/// Core trait for all compute operations
pub trait Compute<S: ?Sized, C: CacheStore<S>> {
    type Input;
    type Output: Cacheable;

    /// Compute the output without caching
    ///
    /// Don't call this method directly. Use `execute` or `execute_many` instead.
    fn raw_execute(&mut self, input: Self::Input) -> Self::Output;

    /// Query the cache with `sig` or compute the output with `input`
    ///
    /// If cache hit, return the cached value. Otherwise, execute the computation and store the
    /// result in the cache.
    ///
    /// This method is to implement caching logic, and useful for implementing `execute` with custom
    /// signature computation logic.
    ///
    /// Don't call this method directly. Use `execute` or `execute_many` instead.
    fn execute_with_sig(
        &mut self,
        sig: &S,
        input: Self::Input,
        cache: &C,
        buffer: &mut <Self::Output as Encodeable>::Buffer,
    ) -> Result<Self::Output> {
        if let Some(cached) = cache.fetch(sig, buffer)? {
            Ok(cached)
        } else {
            let output = self.raw_execute(input);
            cache.store(sig, buffer, &output)?;
            Ok(output)
        }
    }

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
        inputs: impl ParallelIterator<Item = Self::Input>,
    ) -> Result<impl ParallelIterator<Item = Result<Self::Output>>>
    where
        Self: Clone + Sync,
        Self::Output: Send,
    {
        Ok(inputs.map_init(
            || (<Self::Output as Encodeable>::Buffer::init(), self.clone()),
            move |(buffer, c), input| c.execute(input, cache, buffer),
        ))
    }
}

mod single;
pub use single::{PureCompute, StochasiticCompute};

mod multi;
pub use multi::{ChainedSignature, ExpAnalysis};
