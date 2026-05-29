//! Reproducible stochastic compute tasks.
//!
//! This module provides [`StochasticTask`], [`StochasticInput`], and stream seed types for
//! deterministic RNG stream construction.
//!
//! For simulations with multiple random variables, prefer configuring named RNG streams up front.
//! Each variable gets an independent deterministic stream for every input:
//!
//! ```rust
//! # use ssa_workflow::prelude::*;
//! # use rand::{Rng, RngExt};
//! const WAITING_TIME: RandomVariable = RandomVariable::new("ssa-waiting-time-v1");
//! const REACTION_CHOICE: RandomVariable = RandomVariable::new("ssa-reaction-choice-v1");
//!
//! let task = StochasticTask::builder("birth-death-ssa-v1")
//!     .streams([WAITING_TIME, REACTION_CHOICE])
//!     .function(|streams, (initial_cells, max_events): (u32, u32)| {
//!         let [waiting_time_rng, reaction_choice_rng] = streams.as_mut();
//!         let birth_rate = 0.8;
//!         let death_rate = 0.4;
//!         let mut cells = initial_cells.max(1);
//!         let mut time = 0.0;
//!
//!         for _ in 0..max_events {
//!             let birth_propensity = birth_rate * cells as f64;
//!             let death_propensity = death_rate * cells as f64;
//!             let total_propensity = birth_propensity + death_propensity;
//!             if total_propensity == 0.0 {
//!                 break;
//!             }
//!
//!             let u = waiting_time_rng
//!                 .random::<f64>()
//!                 .clamp(f64::MIN_POSITIVE, 1.0);
//!             time += -u.ln() / total_propensity;
//!
//!             let reaction_threshold = reaction_choice_rng.random::<f64>() * total_propensity;
//!             if reaction_threshold < birth_propensity {
//!                 cells = cells.saturating_add(1);
//!             } else {
//!                 cells = cells.saturating_sub(1);
//!             }
//!         }
//!
//!         Ok((cells, time))
//!     })
//!     .cache(ManagedHashCache::<(u32, f64)>::default())
//!     .build()?;
//! # Ok::<_, ssa_workflow::Error>(())
//! ```
//!
//! If the simulation has one natural random trajectory, use the single-stream constructor:
//!
//! ```rust
//! # use ssa_workflow::prelude::*;
//! # use rand::{Rng, RngExt};
//!
//! let task = StochasticTask::builder("birth-death-ssa-v1")
//!     .function(|rng, (initial_cells, max_events): (u32, u32)| {
//!         let birth_rate = 0.8;
//!         let death_rate = 0.4;
//!         let mut cells = initial_cells.max(1);
//!         let mut time = 0.0;
//!
//!         for _ in 0..max_events {
//!             let birth_propensity = birth_rate * cells as f64;
//!             let death_propensity = death_rate * cells as f64;
//!             let total_propensity = birth_propensity + death_propensity;
//!             if total_propensity == 0.0 {
//!                 break;
//!             }
//!
//!             let u = rng.random::<f64>().clamp(f64::MIN_POSITIVE, 1.0);
//!             time += -u.ln() / total_propensity;
//!
//!             let reaction_threshold = rng.random::<f64>() * total_propensity;
//!
//!             if reaction_threshold < birth_propensity {
//!                 cells = cells.saturating_add(1);
//!             } else {
//!                 cells = cells.saturating_sub(1);
//!             }
//!         }
//!
//!         Ok((cells, time))
//!     })
//!     .cache(ManagedHashCache::<(u32, f64)>::default())
//!     .build()?;
//! # Ok::<_, ssa_workflow::Error>(())
//! ```
//!
//! Seed derivation is per stream name:
//!
//! ```text
//! stream: ComputationPath + variable name -> StreamSeed
//! rng:    StreamSeed + encoded_input -> Xoshiro256PlusPlus
//! ```
//!
//! The default single stream is the unnamed stream, so it uses the same seed as
//! `RandomVariable::new("")`.
//!
//! # RNG stability
//!
//! RNG streams are stable across compatible `ssa-workflow` releases for the
//! same [`ComputationId`], [`RandomVariable`] names, canonical input encoding,
//! and `StochasticInput::repetition_index`.
//!
//! Any crate change that alters the RNG stream for the same computation path, stream label, and
//! encoded input is a breaking change. This includes changes to stream
//! derivation, the RNG algorithm, or canonical encoding for [`StochasticInput`].
//!
//! Named streams are independent across random variables. Reordering
//! calls that use different streams from the same [`RngStreams`](crate::compute::RngStreams) bundle
//! does not change the sequence produced by any individual stream.
//!
//! Changes to any seed input change the corresponding RNG stream:
//! computation path, random variable, encoded parameter bytes, or repetition index.
//! The order passed to [`StochasticTaskBuilder::streams`] only changes the order in which streams
//! are delivered to the function; each variable's stream seed is derived independently. Reusing the
//! same variable name twice intentionally produces the same stream twice.
//! Within one stream, random values are consumed sequentially; changing how many values that stream
//! consumes can change later values from the same stream.

use std::marker::PhantomData;

use crate::{
    Compute, Result,
    cache::{Cache, CacheProvider, CanonicalEncode, CloneShared},
    compute::{
        NoFunction,
        stream::{
            NamedStreams, RandomVariable, SeedSource, SingleStream, StreamSeed, StreamSpecSeed,
        },
    },
    identity::{ComputationId, ComputationPath},
};

type BuildStochasticTask<CP, P, O, S, F> =
    StochasticTask<<CP as CacheProvider<O>>::Cache, P, O, S, F>;

/// Input for stochastic computation.
///
/// This wraps a deterministic parameter value (`param`) with a `repetition_index`.
///
/// The pair serves two roles:
///
/// - It is the canonical input used for cache-key construction.
/// - It identifies one reproducible random stream when used with [`StochasticInput`].
///
/// # Encoding
///
/// Canonical encoding is `param` bytes followed by big-endian `repetition_index` bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StochasticInput<P> {
    /// Deterministic model/config input.
    pub param: P,
    /// Repetition index used for random stream derivation.
    pub repetition_index: u64,
}

impl<P> StochasticInput<P> {
    /// Create a stochastic input from `(param, repetition_index)`.
    pub const fn new(param: P, repetition_index: u64) -> Self {
        Self {
            param,
            repetition_index,
        }
    }
}

impl<P> From<(P, u64)> for StochasticInput<P> {
    fn from((param, repetition_index): (P, u64)) -> Self {
        Self {
            param,
            repetition_index,
        }
    }
}

impl<P: CanonicalEncode> CanonicalEncode for StochasticInput<P> {
    const SIZE: usize = P::SIZE + u64::SIZE;

    unsafe fn encode_into(&self, buffer: &mut [u8]) {
        unsafe {
            self.param.encode_into(&mut buffer[..P::SIZE]);
            self.repetition_index
                .encode_into(&mut buffer[P::SIZE..Self::SIZE]);
        }
    }
}

/// Memoized stochastic compute node with reproducible randomness.
///
/// `StochasticTask` is shared by the single-stream and named-stream constructors. The stored
/// `seed` field is generic:
///
/// - `StochasticTask<C, P, O, StreamSeed, F>` stores one [`StreamSeed`] derived from the
///   computation path and the unnamed stream.
/// - `StochasticTask<C, P, O, StreamSeeds<N>, F>` stores pre-derived
///   [`StreamSeeds<N>`](crate::compute::stream::StreamSeeds). Each input uses `StreamSeeds<N> +
///   encoded_input -> RngStreams<N>`.
///
/// Most callers use the constructors and do not name the full generic type.
///
/// # Caching / keyspace contract
///
/// The computation path selects the cache namespace and is also used for RNG seed derivation.
#[derive(Debug)]
pub struct StochasticTask<C, P, O, S, F> {
    path: ComputationPath,
    cache: C,
    seed: S,
    function: F,
    _phantom: PhantomData<(P, O)>,
}

impl StochasticTask<(), (), (), StreamSeed, NoFunction> {
    /// Start a stochastic task builder for a computation id.
    pub fn builder(id: impl Into<ComputationId>) -> StochasticTaskBuilder {
        StochasticTaskBuilder {
            id: id.into(),
            streams: SingleStream,
            function: NoFunction,
            provider: (),
            _phantom: PhantomData,
        }
    }
}

impl<C: CloneShared, P, O, S: Clone, F: Clone> Clone for StochasticTask<C, P, O, S, F> {
    fn clone(&self) -> Self {
        Self {
            path: self.path.clone(),
            cache: self.cache.clone_shared(),
            seed: self.seed.clone(),
            function: self.function.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<C, P, O, F, S> Compute for StochasticTask<C, P, O, S, F>
where
    S: SeedSource,
    F: Fn(&mut S::Rng, P) -> Result<O>,
    C: Cache<O>,
    P: CanonicalEncode,
{
    type Input = StochasticInput<P>;
    type Output = O;

    fn computation_path(&self) -> &ComputationPath {
        &self.path
    }

    fn execute_with_encoded_input(
        &mut self,
        input: Self::Input,
        encoded: &[u8],
    ) -> Result<Self::Output> {
        let param = input.param;
        let Self {
            cache,
            function,
            seed,
            ..
        } = self;

        cache.fetch_or_execute(encoded, || {
            let mut rng = seed.make_rng(encoded);
            function(&mut rng, param)
        })
    }
}

/// Builder for stochastic tasks.
pub struct StochasticTaskBuilder<P = (), O = (), F = NoFunction, S = SingleStream, CP = ()> {
    id: ComputationId,
    streams: S,
    function: F,
    provider: CP,
    _phantom: PhantomData<(P, O)>,
}

impl<P, O, F, S, CP> StochasticTaskBuilder<P, O, F, S, CP> {
    /// Replace the cache provider for this task result.
    ///
    /// If this is not called, the builder uses `()` and executes without caching this task.
    pub fn cache<NextCP>(self, provider: NextCP) -> StochasticTaskBuilder<P, O, F, S, NextCP> {
        StochasticTaskBuilder {
            id: self.id,
            streams: self.streams,
            function: self.function,
            provider,
            _phantom: PhantomData,
        }
    }
}

impl<CP> StochasticTaskBuilder<(), (), NoFunction, SingleStream, CP> {
    /// Replace the default single RNG stream with named streams.
    pub fn streams<const N: usize>(
        self,
        variables: [RandomVariable; N],
    ) -> StochasticTaskBuilder<(), (), NoFunction, NamedStreams<N>, CP> {
        StochasticTaskBuilder {
            id: self.id,
            streams: NamedStreams::new(variables),
            function: self.function,
            provider: self.provider,
            _phantom: PhantomData,
        }
    }
}

impl<P, O, F, S, CP> StochasticTaskBuilder<P, O, F, S, CP>
where
    S: StreamSpecSeed,
{
    /// Replace the compute function for this stochastic task.
    pub fn function<NextP, NextO, NextF>(
        self,
        function: NextF,
    ) -> StochasticTaskBuilder<NextP, NextO, NextF, S, CP>
    where
        NextF: Fn(&mut <<S as StreamSpecSeed>::Seed as SeedSource>::Rng, NextP) -> Result<NextO>,
    {
        StochasticTaskBuilder {
            id: self.id,
            streams: self.streams,
            function,
            provider: self.provider,
            _phantom: PhantomData,
        }
    }
}

impl<P, O, F, S, CP> StochasticTaskBuilder<P, O, F, S, CP>
where
    S: StreamSpecSeed,
    F: Fn(&mut <<S as StreamSpecSeed>::Seed as SeedSource>::Rng, P) -> Result<O>,
    CP: CacheProvider<O>,
    P: CanonicalEncode,
{
    /// Bind the provider and build this stochastic task.
    pub fn build(self) -> Result<BuildStochasticTask<CP, P, O, S::Seed, F>> {
        let path = ComputationPath::root(self.id);
        let cache = self.provider.bind(&path)?;
        let seed = self.streams.derive_seed(&path);
        Ok(StochasticTask {
            path,
            cache,
            seed,
            function: self.function,
            _phantom: PhantomData,
        })
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use rand::{Rng, rngs::Xoshiro256PlusPlus};

    use super::*;
    use crate::cache::memory::ManagedHashCache;

    const MAIN_VARIABLE: RandomVariable = RandomVariable::new("ssa-main-v1");
    const SEGREGATION_VARIABLE: RandomVariable = RandomVariable::new("model-segregation-v1");

    fn test_function(rng: &mut Xoshiro256PlusPlus, _: ()) -> Result<u64> {
        Ok(rng.next_u64())
    }

    mod task_streams {
        use super::*;

        #[test]
        fn test_stochastic_task_reproducible_with_streams() -> Result<()> {
            let mut task = StochasticTask::builder("test-stochastic-streams-v1")
                .streams([MAIN_VARIABLE, SEGREGATION_VARIABLE])
                .function(|rngs, param| {
                    let [main_rng, segregation_rng] = rngs.as_mut();
                    Ok([
                        main_rng.next_u64() ^ param,
                        segregation_rng.next_u64(),
                        main_rng.next_u64(),
                        segregation_rng.next_u64(),
                    ])
                })
                .build()?;

            let output1 = task.execute_one(StochasticInput::new(42, 7))?;
            let output2 = task.execute_one(StochasticInput::new(42, 7))?;

            assert_eq!(output1, output2);
            Ok(())
        }

        #[test]
        fn test_same_input_reuses_same_seed() -> Result<()> {
            let mut task = StochasticTask::builder("experiment-ssa-workflow-test-v1")
                .function(test_function)
                .build()?;
            let output1 = task.execute_one(StochasticInput::new((), 7))?;
            let output2 = task.execute_one(StochasticInput::new((), 7))?;

            assert_eq!(output1, output2);
            Ok(())
        }

        #[test]
        fn test_repetition_index_changes_seed() -> Result<()> {
            let mut task = StochasticTask::builder("experiment-ssa-workflow-test-v1")
                .function(test_function)
                .build()?;
            let output1 = task.execute_one(StochasticInput::new((), 1))?;
            let output2 = task.execute_one(StochasticInput::new((), 2))?;

            assert_ne!(output1, output2);
            Ok(())
        }

        #[test]
        fn test_computation_id_changes_seed() -> Result<()> {
            let mut task1 = StochasticTask::builder("experiment-A-v1")
                .function(test_function)
                .build()?;
            let mut task2 = StochasticTask::builder("experiment-B-v1")
                .function(test_function)
                .build()?;
            let output1 = task1.execute_one(StochasticInput::new((), 7))?;
            let output2 = task2.execute_one(StochasticInput::new((), 7))?;

            assert_ne!(output1, output2);
            Ok(())
        }
    }

    mod execution {
        use super::*;

        #[test]
        fn test_stochastic_cache_isolated_by_compute_instance() -> Result<()> {
            let calls_a = Arc::new(AtomicUsize::new(0));
            let calls_a_clone = calls_a.clone();
            let calls_b = Arc::new(AtomicUsize::new(0));
            let calls_b_clone = calls_b.clone();

            let mut task_a = StochasticTask::builder("experiment-A-v1")
                .function(move |rng, ()| {
                    calls_a_clone.fetch_add(1, Ordering::SeqCst);
                    Ok(rng.next_u64())
                })
                .cache(ManagedHashCache::<u64>::default())
                .build()?;
            let mut task_b = StochasticTask::builder("experiment-B-v1")
                .function(move |rng, ()| {
                    calls_b_clone.fetch_add(1, Ordering::SeqCst);
                    Ok(rng.next_u64())
                })
                .cache(ManagedHashCache::<u64>::default())
                .build()?;

            let input = StochasticInput::new((), 7);

            let output_a1 = task_a.execute_one(input.clone())?;
            let output_b1 = task_b.execute_one(input.clone())?;
            let output_a2 = task_a.execute_one(input.clone())?;
            let output_b2 = task_b.execute_one(input)?;

            assert_eq!(output_a1, output_a2);
            assert_eq!(output_b1, output_b2);
            assert_eq!(calls_a.load(Ordering::SeqCst), 1);
            assert_eq!(calls_b.load(Ordering::SeqCst), 1);
            Ok(())
        }

        #[test]
        fn test_parallel_execution_preserves_seed_reproducibility() -> Result<()> {
            let task1 = StochasticTask::builder("experiment-ssa-workflow-test-v1")
                .function(|rng, ()| Ok(rng.next_u64()))
                .cache(ManagedHashCache::<u64>::default())
                .build()?;
            let task2 = StochasticTask::builder("experiment-ssa-workflow-test-v1")
                .function(|rng, ()| Ok(rng.next_u64()))
                .cache(ManagedHashCache::<u64>::default())
                .build()?;

            let inputs: Vec<_> = (0..128u64).map(|i| StochasticInput::new((), i)).collect();

            let outputs1 = task1.with_inputs(inputs.clone()).collect()?;
            let outputs2 = task2.with_inputs(inputs).collect()?;

            assert_eq!(outputs1, outputs2);
            Ok(())
        }

        #[test]
        fn builder_defaults_to_no_cache() -> Result<()> {
            let call_count = Arc::new(AtomicUsize::new(0));
            let call_count_clone = call_count.clone();

            let mut task = StochasticTask::builder("test-no-cache-stochastic-v1")
                .function(move |rng, ()| {
                    call_count_clone.fetch_add(1, Ordering::SeqCst);
                    Ok(rng.next_u64())
                })
                .build()?;

            let input = StochasticInput::new((), 3);

            assert_eq!(task.execute_one(input.clone())?, task.execute_one(input)?);
            assert_eq!(call_count.load(Ordering::SeqCst), 2);
            Ok(())
        }
    }

    mod input_encoding {
        use super::*;

        #[test]
        fn test_stochastic_input_from_tuple_matches_new_encoding() {
            let from_new = StochasticInput::new(123u64, 9);
            let from_tuple: StochasticInput<u64> = (123u64, 9u64).into();
            let from_fields = StochasticInput {
                param: 123u64,
                repetition_index: 9,
            };

            let mut buffer_new = vec![0u8; StochasticInput::<u64>::SIZE];
            let mut buffer_tuple = vec![0u8; StochasticInput::<u64>::SIZE];
            let mut buffer_fields = vec![0u8; StochasticInput::<u64>::SIZE];

            let encoded_new = unsafe { from_new.encode_with_buffer(&mut buffer_new) };
            let encoded_tuple = unsafe { from_tuple.encode_with_buffer(&mut buffer_tuple) };
            let encoded_fields = unsafe { from_fields.encode_with_buffer(&mut buffer_fields) };

            assert_eq!(encoded_new, encoded_tuple);
            assert_eq!(encoded_new, encoded_fields);
        }
    }
}
