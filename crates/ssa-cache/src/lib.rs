#![doc = include_str!("../README.md")]
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

use std::sync::{Arc, atomic};

use rayon::prelude::*;

pub mod cache;
pub mod error;

pub mod deterministic;
pub mod pipeline;
pub mod stochastic;

use cache::{canonical_encode::CanonicalEncode, codec::CodecEngine, storage::CacheStore};
use error::{Error, Result};

/// Core trait for execution nodes.
///
/// A `Compute` node maps an input to an output, optionally using a cache.
///
/// # Cache key
///
/// `Input` is encoded into canonical bytes via [`CanonicalEncode`]. Those bytes are the cache key.
/// If you change the meaning of your input encoding or the semantics of the computation, treat it
/// as a new keyspace (see the keyspace contract on step/pipeline types).
///
/// # Parallel execution
///
/// [`Compute::execute_many`] accepts a Rayon [`rayon::iter::ParallelIterator`] and returns a
/// parallel iterator of per-item [`Result`] values.
///
/// Internally, execution is structured to minimize allocations:
///
/// - One encode buffer (`Vec<u8>`) per worker, reused across items.
/// - One engine per worker (created via `Self::Engine::default()`), owns its scratch buffer.
/// - One `self.clone()` per worker; implementers should keep `Clone` cheap.
///
/// # Interrupts
///
/// If [`ExecuteOptions`] carries an interrupt signal, it is checked before starting each item.
/// Items already executing are not forcibly cancelled; only pending work is short-circuited with
/// [`Error::Interrupted`].
pub trait Compute {
    /// Input type that can be canonical-encoded into a cache key.
    type Input: CanonicalEncode;
    /// Output type emitted by this node.
    type Output;
    /// Serialization engine used for caching this node's output.
    type Engine: CodecEngine<Self::Output>;

    /// Low-level API for executing with pre-encoded input bytes.
    ///
    /// Most users should prefer [`Self::execute_many`].
    ///
    /// For implementers: `encoded` is the canonical encoding of `input` and is the cache key.
    /// Implementations should avoid re-encoding `input` and use `encoded` directly.
    fn execute_with_encoded_input(
        &mut self,
        input: Self::Input,
        encoded: &[u8],
        engine: &mut Self::Engine,
    ) -> Result<Self::Output>;

    /// Low-level API for executing one input.
    ///
    /// Most users should prefer [`Self::execute_many`].
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
        encode_buffer: &mut [u8],
        engine: &mut Self::Engine,
    ) -> Result<Self::Output> {
        // Safety: The safety is guaranteed by the caller.
        let encoded = unsafe { input.encode_with_buffer(encode_buffer) };
        self.execute_with_encoded_input(input, encoded, engine)
    }

    /// Execute many inputs in parallel.
    ///
    /// `inputs` is a Rayon [`rayon::iter::ParallelIterator`], so work is scheduled across Rayon
    /// worker threads.
    ///
    /// The return value is also a parallel iterator of per-item [`Result`] values, which allows
    /// callers to choose how and when to collect results.
    ///
    /// If `opts` carries an interrupt signal and it becomes `true`, pending items return
    /// [`Error::Interrupted`].
    fn execute_many(
        &self,
        inputs: impl ParallelIterator<Item = Self::Input>,
        opts: ExecuteOptions,
    ) -> Result<impl ParallelIterator<Item = Result<Self::Output>>>
    where
        Self: Clone + Sync,
        Self::Output: Send,
        Self::Engine: Default,
    {
        let signal = opts.signal;
        Ok(inputs.map_init(
            move || {
                (
                    vec![0u8; Self::Input::SIZE],
                    Self::Engine::default(),
                    self.clone(),
                    signal.clone(),
                )
            },
            move |(encode_buffer, engine, c, signal), input| {
                if let Some(signal) = signal
                    && signal.load(atomic::Ordering::Acquire)
                {
                    return Err(Error::Interrupted);
                };

                // Safety: The buffer is initialized with length Self::Input::SIZE.
                unsafe { c.execute(input, encode_buffer, engine) }
            },
        ))
    }
}

/// Optional controls for [`Compute::execute_many`].
///
/// This type is intentionally small and cheap to clone.
#[derive(Debug, Clone, Default)]
pub struct ExecuteOptions {
    signal: Option<Arc<atomic::AtomicBool>>,
}

impl ExecuteOptions {
    /// Create options with an interrupt signal for [`Compute::execute_many`].
    ///
    /// When the signal becomes `true`, pending work items are short-circuited with
    /// [`Error::Interrupted`].
    ///
    /// "Pending" means the work item has not yet started executing on a worker thread.
    pub fn with_interrupt_signal(signal: Arc<atomic::AtomicBool>) -> Self {
        Self {
            signal: Some(signal),
        }
    }
}

pub mod prelude {
    #[cfg(feature = "bitcode")]
    pub use crate::cache::codec::bitcode::Bitcode;
    #[cfg(feature = "lz4")]
    pub use crate::cache::codec::compress::lz4::Lz4;
    #[cfg(feature = "compress")]
    pub use crate::cache::codec::compress::{Compress, CompressedCodec};
    pub use crate::{
        Compute, ExecuteOptions,
        cache::{
            canonical_encode::CanonicalEncode,
            codec::CodecEngine,
            storage::{CacheStore, HashMapStore},
        },
        deterministic::DeterministicStep,
        pipeline::{Pipeline, PipelineExt},
        stochastic::{StochasticInput, StochasticStep},
    };
}
