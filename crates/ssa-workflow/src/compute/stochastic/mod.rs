//! Reproducible stochastic compute tasks.
//!
//! This module provides [`StochasticTask`] and [`StochasticInput`] for reproducible simulations.
//!
//! For simulations with multiple random variables, configure named streams up front. Each name gets
//! its own reproducible RNG for every input:
//!
//! ```rust
//! # use ssa_workflow::prelude::*;
//! # use rand::{Rng, RngExt};
//! let task = StochasticTask::builder("birth-death-ssa-v1")
//!     .streams(["waiting_time", "reaction_choice"])
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
//! If the simulation has one natural random trajectory, keep the default single stream:
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
//! Stream names are stable semantic inputs. Changing a stream name changes only that stream's
//! random sequence.
//!
//! ```text
//! computation path + stream name + encoded input -> RNG
//! ```
//!
//! The default single stream is the unnamed stream, so it uses the same seed as
//! `RandomVariable::new("")`.
//!
//! # RNG stability
//!
//! RNG streams are stable across compatible `ssa-workflow` releases for the same
//! [`ComputationId`], stream names, canonical input encoding, and
//! `StochasticInput::repetition_index`.
//!
//! Named streams are independent across random variables. Reordering
//! calls that use different streams from the same [`RngBundle`](crate::compute::RngBundle) bundle
//! does not change the sequence produced by any individual stream.
//!
//! The order passed to [`StochasticTaskBuilder::streams`] only changes how streams are delivered to
//! the function. Each stream is keyed by name. Reusing the same name twice intentionally produces
//! the same stream twice.
//! Within one stream, random values are consumed sequentially; changing how many values that stream
//! consumes can change later values from the same stream.

use std::marker::PhantomData;

use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use crate::{
    BatchExecution, Compute, Result,
    cache::{Cache, CacheProvider, CanonicalEncode, CloneShared},
    compute::{
        NoFunction,
        stream::{MultiStreams, SeedSource, SingleStream, StreamSeed, StreamSpec},
    },
    identity::{ComputationId, ComputationPath},
};

type BuildStochasticTask<CP, P, O, S, F> =
    StochasticTask<<CP as CacheProvider<O>>::Cache, P, O, S, F>;

/// Input for one stochastic execution.
///
/// This wraps a deterministic parameter value (`param`) with a `repetition_index`.
///
/// The pair serves two roles:
///
/// - It is the canonical input used for cache-key construction.
/// - It selects one reproducible stochastic repetition.
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

/// Convenience methods for computations keyed by [`StochasticInput`].
///
/// This trait is implemented for every [`Compute`] node, including stochastic tasks and
/// deterministic transforms built from stochastic tasks. Import it from the prelude to run the
/// common "same deterministic input, many stochastic repetitions" batch without manually building
/// `StochasticInput` values.
pub trait StochasticComputeExt: Compute {
    /// Start a batch execution for `repetitions` stochastic repetitions of one deterministic input.
    ///
    /// The batch uses `0..repetitions` as its indexed parallel input range and converts each item
    /// to `StochasticInput::new(param.clone(), index as u64)`. Collected outputs therefore
    /// remain in repetition-index order while the stochastic identity continues to use a
    /// platform-stable `u64` repetition id.
    fn with_repeated_input<P>(
        &self,
        param: P,
        repetitions: usize,
    ) -> BatchExecution<'_, Self, impl IndexedParallelIterator<Item = StochasticInput<P>>>
    where
        Self: Sized + Compute<Input = StochasticInput<P>>,
        P: Clone + Send,
    {
        self.with_inputs(
            (0..repetitions)
                .into_par_iter()
                .map_with(param, |param, index| {
                    StochasticInput::new(param.clone(), index as u64)
                }),
        )
    }
}

impl<T: Compute> StochasticComputeExt for T {}

/// Memoized stochastic task with reproducible randomness.
///
/// Create a task with [`StochasticTask::builder`], attach a function and cache provider, then call
/// `build`.
///
/// # Caching / keyspace contract
///
/// The computation id selects the cache namespace and the reproducible RNG stream family. Bump the
/// id when the task's output semantics change.
#[derive(Debug)]
pub struct StochasticTask<C, P, O, S, F> {
    path: ComputationPath,
    cache: C,
    seed: S,
    function: F,
    _phantom: PhantomData<(P, O)>,
}

impl StochasticTask<(), (), (), StreamSeed, NoFunction> {
    /// Start a stochastic task builder for a stable computation id.
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

impl<C, P, O, S, F> Compute for StochasticTask<C, P, O, S, F>
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
    /// Replace the default single stream with named streams.
    pub fn streams<const N: usize, I: Into<MultiStreams<N>>>(
        self,
        variables: I,
    ) -> StochasticTaskBuilder<(), (), NoFunction, MultiStreams<N>, CP> {
        StochasticTaskBuilder {
            id: self.id,
            streams: variables.into(),
            function: self.function,
            provider: self.provider,
            _phantom: PhantomData,
        }
    }
}

impl<P, O, F, S: StreamSpec, CP> StochasticTaskBuilder<P, O, F, S, CP> {
    /// Replace the compute function for this stochastic task.
    pub fn function<NextP, NextO, NextF>(
        self,
        function: NextF,
    ) -> StochasticTaskBuilder<NextP, NextO, NextF, S, CP>
    where
        NextF: Fn(&mut <S::Seed as SeedSource>::Rng, NextP) -> Result<NextO>,
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
    S: StreamSpec,
    F: Fn(&mut <S::Seed as SeedSource>::Rng, P) -> Result<O>,
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

    fn test_function(rng: &mut Xoshiro256PlusPlus, _: ()) -> Result<u64> {
        Ok(rng.next_u64())
    }

    mod task_streams {
        use super::*;

        #[test]
        fn test_stochastic_task_reproducible_with_streams() -> Result<()> {
            let mut task = StochasticTask::builder("test-stochastic-streams-v1")
                .streams(["main", "segregation"])
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
