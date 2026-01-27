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
pub use cache::codec::bitcode_codec::BitcodeCodec;
pub use cache::{
    codec::{Codec, CodecEngine, Decode, DefaultCodec, Encode},
    storage::{CacheStore, HashMapStore},
};

/// Core trait for all compute operations
pub trait Compute<C, E: CodecEngine> {
    type Input;
    type Output;

    /// Execute the computation or fetch from cache
    ///
    /// If cache hit, return the cached value. Otherwise, execute the computation and store the
    /// result in the cache.
    fn execute(&mut self, input: Self::Input, cache: &C, engine: &mut E) -> Result<Self::Output>;

    /// Execute many computations in parallel
    fn execute_many(
        &self,
        inputs: impl ParallelIterator<Item = Self::Input>,
        cache: &C,
        opts: ExecuteOptions<E>,
    ) -> Result<impl ParallelIterator<Item = Result<Self::Output>>>
    where
        C: Sync,
        Self: Clone + Sync,
        Self::Output: Send,
        E: Send,
    {
        let signal = opts.signal;
        Ok(inputs.map_init(
            move || (E::default(), self.clone(), signal.clone()),
            move |(engine, c, signal), input| {
                if let Some(signal) = signal
                    && signal.load(Ordering::Acquire)
                {
                    return Err(Error::Interrupted);
                };

                c.execute(input, cache, engine)
            },
        ))
    }
}

#[derive(Debug, Clone, Default)]
pub struct ExecuteOptions<E> {
    signal: Option<Arc<AtomicBool>>,
    _marker: std::marker::PhantomData<E>,
}

impl<E> ExecuteOptions<E> {
    pub fn with_interrupt_signal(signal: Arc<AtomicBool>) -> Self {
        Self {
            signal: Some(signal),
            _marker: std::marker::PhantomData,
        }
    }
}

mod single;
pub use single::{PureCompute, StochasiticCompute};

mod multi;
pub use multi::ExpAnalysis;
