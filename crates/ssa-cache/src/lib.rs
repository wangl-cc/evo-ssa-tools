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
    canonical_encode::CanonicalEncode,
    codec::{Codec, CodecBuffer, DefaultCodec},
    storage::{CacheStore, HashMapStore},
};

/// Core trait for all compute operations
pub trait Compute<C> {
    type Input: CanonicalEncode;
    type Output: Codec;

    /// Execute the computation or fetch from cache
    ///
    /// If cache hit, return the cached value. Otherwise, execute the computation and store the
    /// result in the cache.
    fn execute(
        &mut self,
        input: Self::Input,
        cache: &C,
        encode_buffer: &mut [u8],
        codex_buffer: &mut <Self::Output as Codec>::Buffer,
    ) -> Result<Self::Output>;

    /// Execute many computations in parallel
    fn execute_many(
        &self,
        inputs: impl ParallelIterator<Item = Self::Input>,
        cache: &C,
        opts: ExecuteOptions,
    ) -> Result<impl ParallelIterator<Item = Result<Self::Output>>>
    where
        C: Sync,
        Self: Clone + Sync,
        Self::Output: Send,
    {
        let signal = opts.signal;
        Ok(inputs.map_init(
            move || {
                (
                    vec![0u8; Self::Input::SIZE],
                    <<Self::Output as Codec>::Buffer as CodecBuffer>::init(),
                    self.clone(),
                    signal.clone(),
                )
            },
            move |(encode_buffer, codec_buffer, c, signal), input| {
                if let Some(signal) = signal
                    && signal.load(Ordering::Acquire)
                {
                    return Err(Error::Interrupted);
                };

                c.execute(input, cache, encode_buffer, codec_buffer)
            },
        ))
    }
}

#[derive(Debug, Clone, Default)]
pub struct ExecuteOptions {
    signal: Option<Arc<AtomicBool>>,
}

impl ExecuteOptions {
    pub fn with_interrupt_signal(signal: Arc<AtomicBool>) -> Self {
        Self {
            signal: Some(signal),
        }
    }
}

mod single;
pub use single::{PureCompute, StochasiticCompute};

mod multi;
pub use multi::ExpAnalysis;
