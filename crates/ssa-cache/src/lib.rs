#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

use std::sync::{Arc, atomic};

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

    fn execute_with_sig(
        &mut self,
        input: Self::Input,
        input_signature: &[u8],
        cache: &C,
        codec_buffer: &mut <Self::Output as Codec>::Buffer,
    ) -> Result<Self::Output>;

    /// Execute the computation or fetch from cache
    ///
    /// If cache hit, return the cached value. Otherwise, execute the computation and store the
    /// result in the cache.
    ///
    /// # Safety
    ///
    /// The buffer must be with length at least `Self::Input::SIZE`.
    /// And implementation should only access buffer[..Self::Input::SIZE].
    ///
    /// See [`CanonicalEncode`] for more details.
    unsafe fn execute(
        &mut self,
        input: Self::Input,
        cache: &C,
        encode_buffer: &mut [u8],
        codec_buffer: &mut <Self::Output as Codec>::Buffer,
    ) -> Result<Self::Output> {
        // Safety: The safety is guaranteed by the caller.
        let input_signature = unsafe { input.encode_with_buffer(encode_buffer) };
        self.execute_with_sig(input, input_signature, cache, codec_buffer)
    }

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
                    && signal.load(atomic::Ordering::Acquire)
                {
                    return Err(Error::Interrupted);
                };

                // Safety: The buffer is initialized with length Self::Input::SIZE.
                unsafe { c.execute(input, cache, encode_buffer, codec_buffer) }
            },
        ))
    }
}

#[derive(Debug, Clone, Default)]
pub struct ExecuteOptions {
    signal: Option<Arc<atomic::AtomicBool>>,
}

impl ExecuteOptions {
    pub fn with_interrupt_signal(signal: Arc<atomic::AtomicBool>) -> Self {
        Self {
            signal: Some(signal),
        }
    }
}

mod single;
/// Deterministic single-stage compute with cache.
pub use single::SingleStep;

mod stochastic;
/// Deterministic stochastic compute with per-input stream derivation.
pub use stochastic::{StochasticInput, StochasticStep};

mod multi;
/// Multi-stage compute pipeline with per-stage cache.
pub use multi::Pipeline;
