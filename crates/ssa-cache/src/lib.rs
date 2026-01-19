#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use rayon::prelude::*;

mod error;
pub use error::{Error, Result};

mod cache;
#[cfg(feature = "bitcode")]
pub use cache::codec::BitcodeCodec;
pub use cache::{
    codec::{CodecEngine, Decode, Encode},
    storage::{CacheStore, HashMapStore},
};

/// Core trait for all compute operations
pub trait Compute<C> {
    type Engine: CodecEngine;
    type Input;
    type Output;

    /// Execute the computation or fetch from cache
    ///
    /// If cache hit, return the cached value. Otherwise, execute the computation and store the
    /// result in the cache.
    fn execute(
        &mut self,
        input: Self::Input,
        cache: &C,
        engine: &mut Self::Engine,
    ) -> Result<Self::Output>;

    /// Execute many computations in parallel
    fn execute_many(
        &self,
        cache: &C,
        interrupted: Arc<AtomicBool>,
        inputs: impl ParallelIterator<Item = Self::Input>,
    ) -> Result<impl ParallelIterator<Item = Result<Self::Output>>>
    where
        C: Sync,
        Self: Clone + Sync,
        Self::Output: Send,
        Self::Engine: Send,
    {
        Ok(inputs.map_init(
            || (Self::Engine::default(), self.clone()),
            move |(engine, c), input| {
                if interrupted.load(Ordering::Acquire) {
                    return Err(Error::Interrupted);
                };

                c.execute(input, cache, engine)
            },
        ))
    }
}

pub fn fetch_or_execute<C, E, O, F>(sig: &[u8], cache: &C, engine: &mut E, execute: F) -> Result<O>
where
    C: CacheStore<E>,
    E: Encode<O> + Decode<O>,
    F: FnOnce() -> Result<O>,
{
    if let Some(cached) = cache.fetch(sig, engine)? {
        Ok(cached)
    } else {
        let output = execute()?;
        cache.store(sig, engine, &output)?;
        Ok(output)
    }
}

mod single;

pub type PureCompute<I, O, F, C = BitcodeCodec> = single::PureCompute<I, O, F, C>;
pub type StochasiticCompute<I, O, F, G, C = BitcodeCodec> =
    single::StochasiticCompute<I, O, F, G, C>;

mod multi;
pub type ExpAnalysis<I, M, O, E, A, G, C = BitcodeCodec> = multi::ExpAnalysis<I, M, O, E, A, G, C>;
