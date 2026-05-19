//! Reproducible stochastic compute steps.
//!
//! This module provides [`StochasticStep`], [`StochasticInput`], and seed/model types for
//! deterministic RNG stream construction.
//!
//! For simulations with multiple random variables, prefer configuring named RNG streams up front.
//! Each variable gets an independent deterministic stream for every input:
//!
//! ```rust
//! # use ssa_pipeline::prelude::*;
//! # use rand::{Rng, RngExt};
//! # #[cfg(feature = "bitcode")]
//! # {
//! const MODEL: SimulationModel = SimulationModel::new("birth-death-ssa/v1");
//! const WAITING_TIME: RandomVariable = RandomVariable::new("ssa/waiting-time/v1");
//! const REACTION_CHOICE: RandomVariable = RandomVariable::new("ssa/reaction-choice/v1");
//!
//! let _step = StochasticStep::new_with_streams(
//!     DefaultHashMapStore::default(),
//!     MODEL,
//!     [WAITING_TIME, REACTION_CHOICE],
//!     |streams, (initial_cells, max_events): (u32, u32)| {
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
//!     },
//!     Bitcode06::default,
//! );
//! # }
//! ```
//!
//! If the simulation has one natural random trajectory, use the single-stream constructor:
//!
//! ```rust
//! # use ssa_pipeline::prelude::*;
//! # use rand::{Rng, RngExt};
//! # #[cfg(feature = "bitcode")]
//! # {
//! const MODEL: SimulationModel = SimulationModel::new("birth-death-ssa/v1");
//!
//! let _step = StochasticStep::new(
//!     DefaultHashMapStore::default(),
//!     MODEL,
//!     |rng, (initial_cells, max_events): (u32, u32)| {
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
//!     },
//!     Bitcode06::default,
//! );
//! # }
//! ```
//!
//! Seed derivation differs by mode:
//!
//! ```text
//! single stream: SimulationModel + internal single-stream variable -> StreamSeed
//! single stream: StreamSeed + encoded_input -> Xoshiro256PlusPlus
//!
//! named streams: SimulationModel + RandomVariable -> StreamSeed
//! named streams: StreamSeed + encoded_input -> Xoshiro256PlusPlus
//! ```
//!
//! # RNG stability
//!
//! RNG streams are stable across compatible `ssa-pipeline` releases for the
//! same [`SimulationModel`], [`RandomVariable`] list, canonical input encoding,
//! and `StochasticInput::repetition_index`.
//!
//! Any crate change that alters the RNG stream for the same stream identity and
//! encoded input is a breaking change. This includes changes to stream
//! derivation, the RNG algorithm, the crate-internal single-stream variable used
//! by [`StochasticStep::new`], or canonical encoding for [`StochasticInput`].
//!
//! Named streams are independent across random variables. Reordering
//! calls that use different streams from the same [`RngStreams`] bundle
//! does not change the sequence produced by any individual stream.
//!
//! Changes to any stream identity input change the corresponding RNG stream:
//! simulation model, random variable, encoded parameter bytes, repetition index,
//! or the variable order passed to [`StochasticStep::new_with_streams`].
//! Within one stream, random values are consumed sequentially; changing how many values that stream
//! consumes can change later values from the same stream.

use std::marker::PhantomData;

use rand::rngs::Xoshiro256PlusPlus;

pub mod seed;

pub use seed::{RandomVariable, RngStreams, SimulationModel, StreamSeed, StreamSeeds};

use crate::{
    CacheStore, CanonicalEncode, Compute, Result,
    cache::{
        codec::{CodecEngine, EngineFactory},
        storage::WorkerForkStore,
    },
};

/// Input for stochastic computation.
///
/// This wraps a deterministic parameter value (`param`) with a `repetition_index`.
///
/// The pair serves two roles:
///
/// - It is the canonical input used for cache-key construction.
/// - It identifies one reproducible random stream when used with [`StochasticStep`].
///
/// # Encoding
///
/// Canonical encoding is `param` bytes followed by big-endian `repetition_index` bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StochasticInput<P> {
    /// Deterministic model/config input.
    param: P,
    /// Repetition index used for random stream derivation.
    repetition_index: u64,
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
/// `StochasticStep` is shared by the single-stream and named-stream constructors. The stored
/// `seed` field is generic:
///
/// - `StochasticStep<C, P, O, StreamSeed, F, EF>` is created by [`StochasticStep::new`] and stores
///   one [`StreamSeed`] derived from the simulation model and the internal single-stream variable.
/// - `StochasticStep<C, P, O, StreamSeeds<N>, F, EF>` is created by
///   [`StochasticStep::new_with_streams`] and stores pre-derived [`StreamSeeds<N>`]. Each input
///   uses `StreamSeeds<N> + encoded_input -> RngStreams<N>`.
///
/// Most callers use the constructors and do not name the full generic type.
///
/// # Caching / keyspace contract
///
/// The provided `cache` is treated as this step's private keyspace.
/// Reusing the same underlying keyspace for different stochastic steps is not supported and may
/// return cached values written by another step for the same input bytes.
/// Keyspace compatibility is caller-managed: when seed derivation, stochastic algorithm, or
/// encoding semantics change in an incompatible way, use a new keyspace.
///
/// Note: the simulation model only affects random stream derivation. It is not used as a
/// cache-key namespace.
#[derive(Debug)]
pub struct StochasticStep<C, P, O, S, F, EF> {
    cache: C,
    seed: S,
    function: F,
    engine_factory: EF,
    _phantom: PhantomData<(P, O)>,
}

impl<C, P, O, F, EF> StochasticStep<C, P, O, StreamSeed, F, EF>
where
    F: Fn(&mut Xoshiro256PlusPlus, P) -> Result<O>,
    EF: EngineFactory,
    EF::Engine: CodecEngine<O>,
{
    /// Create a stochastic step with a single RNG stream.
    ///
    /// `cache` should be dedicated to this step (see [`StochasticStep`] cache keyspace contract).
    /// `model` affects seed derivation only; it does not namespace cache keys.
    pub fn new(cache: C, model: SimulationModel, function: F, engine_factory: EF) -> Self {
        Self {
            cache,
            seed: model.derive_single_stream_seed(),
            function,
            engine_factory,
            _phantom: PhantomData,
        }
    }
}

impl<C, P, O, F, EF, const N: usize> StochasticStep<C, P, O, StreamSeeds<N>, F, EF>
where
    F: Fn(&mut RngStreams<N>, P) -> Result<O>,
    EF: EngineFactory,
    EF::Engine: CodecEngine<O>,
{
    /// Create a stochastic step whose function receives configured RNG streams.
    ///
    /// `cache` should be dedicated to this step (see [`StochasticStep`] cache keyspace contract).
    /// `model` affects seed derivation only; it does not namespace cache keys.
    pub fn new_with_streams(
        cache: C,
        model: SimulationModel,
        variables: [RandomVariable; N],
        function: F,
        engine_factory: EF,
    ) -> Self {
        Self {
            cache,
            seed: model.derive_stream_seeds(variables),
            function,
            engine_factory,
            _phantom: PhantomData,
        }
    }
}

impl<C: WorkerForkStore, P, O, S: Clone, F: Clone, EF: Clone> Clone
    for StochasticStep<C, P, O, S, F, EF>
{
    fn clone(&self) -> Self {
        Self {
            cache: self.cache.fork_store(),
            seed: self.seed.clone(),
            function: self.function.clone(),
            engine_factory: self.engine_factory.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<C, P, O, F, EF> Compute for StochasticStep<C, P, O, StreamSeed, F, EF>
where
    F: Fn(&mut Xoshiro256PlusPlus, P) -> Result<O>,
    C: CacheStore,
    P: CanonicalEncode,
    EF: EngineFactory,
    EF::Engine: CodecEngine<O>,
{
    type Engine = EF::Engine;
    type Input = StochasticInput<P>;
    type Output = O;

    fn make_engine(&self) -> Self::Engine {
        self.engine_factory.make_engine()
    }

    fn execute_with_encoded_input(
        &mut self,
        input: Self::Input,
        encoded: &[u8],
        engine: &mut Self::Engine,
    ) -> Result<Self::Output> {
        let param = input.param;
        let Self {
            cache,
            function,
            seed,
            ..
        } = self;

        cache.fetch_or_execute(encoded, engine, |_| {
            let mut rng = seed.make_stream(encoded);
            function(&mut rng, param)
        })
    }
}

impl<C, P, O, F, EF, const N: usize> Compute for StochasticStep<C, P, O, StreamSeeds<N>, F, EF>
where
    F: Fn(&mut RngStreams<N>, P) -> Result<O>,
    C: CacheStore,
    P: CanonicalEncode,
    EF: EngineFactory,
    EF::Engine: CodecEngine<O>,
{
    type Engine = EF::Engine;
    type Input = StochasticInput<P>;
    type Output = O;

    fn make_engine(&self) -> Self::Engine {
        self.engine_factory.make_engine()
    }

    fn execute_with_encoded_input(
        &mut self,
        input: Self::Input,
        encoded: &[u8],
        engine: &mut Self::Engine,
    ) -> Result<Self::Output> {
        let param = input.param;
        let cache = &self.cache;
        let function = &self.function;
        let seed = &self.seed;

        cache.fetch_or_execute(encoded, engine, |_| {
            let mut streams = seed.make_streams(encoded);
            function(&mut streams, param)
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

    use rand::Rng;
    use rayon::prelude::*;

    use super::*;
    use crate::{
        cache::{codec::fixtures::FixtureEngine, storage::DefaultHashMapStore},
        prelude::*,
        test_utils::execute_one,
    };

    const TEST_MODEL: SimulationModel = SimulationModel::new("experiment/ssa-pipeline-test/v1");
    const MAIN_VARIABLE: RandomVariable = RandomVariable::new("ssa/main/v1");
    const SEGREGATION_VARIABLE: RandomVariable = RandomVariable::new("model/segregation/v1");
    const MUTATION_VARIABLE: RandomVariable = RandomVariable::new("model/mutation/v1");

    #[test]
    fn test_model_and_random_variable_names_are_stable() {
        let model = SimulationModel::new("experiment/test/v1");
        let variable = RandomVariable::new("test/stream/v1");

        assert_eq!(model.as_str(), "experiment/test/v1");
        assert_eq!(variable.as_str(), "test/stream/v1");
        assert_eq!(model.to_string(), "experiment/test/v1");
        assert_eq!(variable.to_string(), "test/stream/v1");
    }

    #[test]
    fn test_stream_seed_debug_output_is_redacted() {
        let stream_seed = TEST_MODEL.derive_stream_seed(MAIN_VARIABLE);

        let stream_debug = format!("{stream_seed:?}");

        assert_eq!(stream_debug, "StreamSeed { .. }");
        assert!(!stream_debug.contains("bytes"));
    }

    #[test]
    fn test_stream_seed_streams_are_stable_and_isolated() {
        let segregation_seed = TEST_MODEL.derive_stream_seed(SEGREGATION_VARIABLE);
        let mutation_seed = TEST_MODEL.derive_stream_seed(MUTATION_VARIABLE);
        let mut segregation1 = segregation_seed.make_stream(b"input-A");
        let mut segregation2 = segregation_seed.make_stream(b"input-A");
        let mut mutation = mutation_seed.make_stream(b"input-A");

        assert_eq!(segregation1.next_u64(), segregation2.next_u64());
        assert_ne!(segregation1.next_u64(), mutation.next_u64());
    }

    #[test]
    fn test_stream_seed_input_bytes_change_stream() {
        let seed = TEST_MODEL.derive_stream_seed(SEGREGATION_VARIABLE);
        let mut rng_a = seed.make_stream(b"input-A");
        let mut rng_b = seed.make_stream(b"input-B");

        assert_ne!(rng_a.next_u64(), rng_b.next_u64());
    }

    #[test]
    fn test_single_stream_seed_known_sequence() {
        let seed = TEST_MODEL.derive_single_stream_seed();
        let mut rng = seed.make_stream(b"input-A");

        assert_eq!(rng.next_u64(), 5_883_700_003_951_674_005);
        assert_eq!(rng.next_u64(), 8_534_955_113_658_070_652);
    }

    #[test]
    fn test_stream_seed_known_sequence() {
        let seed = TEST_MODEL.derive_stream_seed(SEGREGATION_VARIABLE);
        let mut rng = seed.make_stream(b"input-A");

        assert_eq!(rng.next_u64(), 8_696_119_191_897_611_845);
        assert_eq!(rng.next_u64(), 11_319_017_350_160_744_450);
    }

    #[test]
    fn test_stream_seed_bundle_allows_duplicate_variables() {
        let seeds = TEST_MODEL.derive_stream_seeds([SEGREGATION_VARIABLE, SEGREGATION_VARIABLE]);
        let mut streams = seeds.make_streams(b"input-A");
        let [left, right] = streams.as_mut();

        assert_eq!(left.next_u64(), right.next_u64());
    }

    #[test]
    fn test_stream_seed_bundle_accessors_preserve_order() {
        let seeds = TEST_MODEL.derive_stream_seeds([SEGREGATION_VARIABLE, MUTATION_VARIABLE]);
        let [segregation_seed_ref, mutation_seed_ref] = seeds.as_ref();
        let mut segregation_rng = segregation_seed_ref.make_stream(b"input-A");
        let mut mutation_rng = mutation_seed_ref.make_stream(b"input-A");

        let [segregation_seed, mutation_seed] = seeds.into_inner();
        let mut owned_segregation_rng = segregation_seed.make_stream(b"input-A");
        let mut owned_mutation_rng = mutation_seed.make_stream(b"input-A");

        assert_eq!(segregation_rng.next_u64(), owned_segregation_rng.next_u64());
        assert_eq!(mutation_rng.next_u64(), owned_mutation_rng.next_u64());
    }

    #[test]
    fn test_single_stream_seed_is_isolated_from_named_stream_seed() {
        let single_seed = TEST_MODEL.derive_single_stream_seed();
        let stream_seed = TEST_MODEL.derive_stream_seed(MAIN_VARIABLE);
        let mut single_rng = single_seed.make_stream(b"input-A");
        let mut stream_rng = stream_seed.make_stream(b"input-A");

        assert_ne!(single_rng.next_u64(), stream_rng.next_u64());
    }

    #[test]
    fn test_stream_seed_bundle_supports_multiple_mutable_rngs() {
        let seeds = TEST_MODEL.derive_stream_seeds([SEGREGATION_VARIABLE, MUTATION_VARIABLE]);
        let mut streams = seeds.make_streams(b"input-A");
        let [segregation_rng, mutation_rng] = streams.as_mut();

        let segregation_value = segregation_rng.next_u64();
        let mutation_value = mutation_rng.next_u64();

        assert_ne!(segregation_value, mutation_value);
    }

    #[test]
    fn test_stochastic_stream_bundle_into_inner_preserves_order() {
        let seeds = TEST_MODEL.derive_stream_seeds([SEGREGATION_VARIABLE, MUTATION_VARIABLE]);
        let mut streams = seeds.make_streams(b"input-A");
        let [segregation_rng, mutation_rng] = streams.as_mut();

        let segregation_value = segregation_rng.next_u64();
        let mutation_value = mutation_rng.next_u64();

        let streams = seeds.make_streams(b"input-A");
        let [mut owned_segregation_rng, mut owned_mutation_rng] = streams.into_inner();

        assert_eq!(segregation_value, owned_segregation_rng.next_u64());
        assert_eq!(mutation_value, owned_mutation_rng.next_u64());
    }

    #[test]
    fn test_stochastic_step_reproducible_with_streams() -> Result<()> {
        let mut step = StochasticStep::new_with_streams(
            (),
            TEST_MODEL,
            [MAIN_VARIABLE, SEGREGATION_VARIABLE],
            |rngs, param| {
                let [main_rng, segregation_rng] = rngs.as_mut();
                Ok([
                    main_rng.next_u64() ^ param,
                    segregation_rng.next_u64(),
                    main_rng.next_u64(),
                    segregation_rng.next_u64(),
                ])
            },
            FixtureEngine::default,
        );

        let output1 = execute_one(&mut step, StochasticInput::new(42, 7))?;
        let output2 = execute_one(&mut step, StochasticInput::new(42, 7))?;

        assert_eq!(output1, output2);
        Ok(())
    }

    #[test]
    fn test_stochastic_reproducible_with_owned_cache() -> Result<()> {
        let mut step = StochasticStep::new(
            (),
            TEST_MODEL,
            |rng, param| {
                Ok([
                    rng.next_u64() ^ param,
                    rng.next_u64(),
                    rng.next_u64(),
                    rng.next_u64(),
                ])
            },
            FixtureEngine::default,
        );
        let output1 = execute_one(&mut step, StochasticInput::new(42, 7))?;
        let output2 = execute_one(&mut step, StochasticInput::new(42, 7))?;

        assert_eq!(output1, output2);
        Ok(())
    }

    #[test]
    fn test_stochastic_diff_repetition_index_changes_stream() -> Result<()> {
        let mut step = StochasticStep::new(
            (),
            TEST_MODEL,
            |rng, param| {
                Ok([
                    rng.next_u64() ^ param,
                    rng.next_u64(),
                    rng.next_u64(),
                    rng.next_u64(),
                ])
            },
            FixtureEngine::default,
        );
        let output1 = execute_one(&mut step, StochasticInput::new(42, 1))?;
        let output2 = execute_one(&mut step, StochasticInput::new(42, 2))?;

        assert_ne!(output1, output2);
        Ok(())
    }

    #[test]
    fn test_stochastic_diff_simulation_model_isolated() -> Result<()> {
        let mut step1 = StochasticStep::new(
            (),
            SimulationModel::new("experiment/A/v1"),
            |rng, param| {
                Ok([
                    rng.next_u64() ^ param,
                    rng.next_u64(),
                    rng.next_u64(),
                    rng.next_u64(),
                ])
            },
            FixtureEngine::default,
        );
        let mut step2 = StochasticStep::new(
            (),
            SimulationModel::new("experiment/B/v1"),
            |rng, param| {
                Ok([
                    rng.next_u64() ^ param,
                    rng.next_u64(),
                    rng.next_u64(),
                    rng.next_u64(),
                ])
            },
            FixtureEngine::default,
        );
        let output1 = execute_one(&mut step1, StochasticInput::new(42, 7))?;
        let output2 = execute_one(&mut step2, StochasticInput::new(42, 7))?;

        assert_ne!(output1, output2);
        Ok(())
    }

    #[test]
    fn test_stochastic_cache_isolated_by_compute_instance() -> Result<()> {
        let calls_a = Arc::new(AtomicUsize::new(0));
        let calls_a_clone = calls_a.clone();
        let calls_b = Arc::new(AtomicUsize::new(0));
        let calls_b_clone = calls_b.clone();

        let mut step_a = StochasticStep::new(
            DefaultHashMapStore::default(),
            SimulationModel::new("experiment/A/v1"),
            move |rng, param| {
                calls_a_clone.fetch_add(1, Ordering::SeqCst);
                Ok(rng.next_u64() ^ param)
            },
            FixtureEngine::default,
        );
        let mut step_b = StochasticStep::new(
            DefaultHashMapStore::default(),
            SimulationModel::new("experiment/B/v1"),
            move |rng, param| {
                calls_b_clone.fetch_add(1, Ordering::SeqCst);
                Ok(rng.next_u64() ^ param)
            },
            FixtureEngine::default,
        );

        let input = StochasticInput::new(42, 7);

        let output_a1 = execute_one(&mut step_a, input.clone())?;
        let output_b1 = execute_one(&mut step_b, input.clone())?;
        let output_a2 = execute_one(&mut step_a, input.clone())?;
        let output_b2 = execute_one(&mut step_b, input)?;

        assert_eq!(output_a1, output_a2);
        assert_eq!(output_b1, output_b2);
        assert_eq!(calls_a.load(Ordering::SeqCst), 1);
        assert_eq!(calls_b.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[test]
    fn test_stochastic_execute_many_parallel_reproducible() -> Result<()> {
        let step1 = StochasticStep::new(
            DefaultHashMapStore::default(),
            TEST_MODEL,
            |rng, param| Ok(rng.next_u64() ^ param),
            FixtureEngine::default,
        );
        let step2 = StochasticStep::new(
            DefaultHashMapStore::default(),
            TEST_MODEL,
            |rng, param| Ok(rng.next_u64() ^ param),
            FixtureEngine::default,
        );

        let inputs: Vec<_> = (0..128u64)
            .map(|i| StochasticInput::new(i, i % 8))
            .collect();

        let outputs1 = step1
            .execute_many(inputs.clone().into_par_iter(), ExecuteOptions::default())?
            .collect::<Result<Vec<u64>>>()?;
        let outputs2 = step2
            .execute_many(inputs.into_par_iter(), ExecuteOptions::default())?
            .collect::<Result<Vec<u64>>>()?;

        assert_eq!(outputs1, outputs2);
        Ok(())
    }

    #[test]
    fn test_stochastic_pipeline_integration() -> Result<()> {
        let stage1_calls = Arc::new(AtomicUsize::new(0));
        let stage1_calls_clone = stage1_calls.clone();
        let stage2_calls = Arc::new(AtomicUsize::new(0));
        let stage2_calls_clone = stage2_calls.clone();

        let stage1 = StochasticStep::new(
            DefaultHashMapStore::default(),
            TEST_MODEL,
            move |rng, param| {
                stage1_calls_clone.fetch_add(1, Ordering::SeqCst);
                Ok((rng.next_u64() as usize) ^ param)
            },
            FixtureEngine::default,
        );

        let pipeline = Pipeline::new(
            stage1,
            DefaultHashMapStore::default(),
            move |intermediate| {
                stage2_calls_clone.fetch_add(1, Ordering::SeqCst);
                Ok(intermediate + 10)
            },
        );

        let inputs: Vec<_> = (0..20usize)
            .map(|i| StochasticInput::new(i, (i % 4) as u64))
            .collect();

        let outputs1 = pipeline
            .execute_many(inputs.clone().into_par_iter(), ExecuteOptions::default())?
            .collect::<Result<Vec<usize>>>()?;
        let outputs2 = pipeline
            .execute_many(inputs.into_par_iter(), ExecuteOptions::default())?
            .collect::<Result<Vec<usize>>>()?;

        assert_eq!(outputs1, outputs2);
        assert_eq!(stage1_calls.load(Ordering::SeqCst), 20);
        assert_eq!(stage2_calls.load(Ordering::SeqCst), 20);
        Ok(())
    }

    #[test]
    fn test_stochastic_input_from_tuple_matches_new_encoding() {
        let from_new = StochasticInput::new(123u64, 9);
        let from_tuple: StochasticInput<u64> = (123u64, 9u64).into();

        let mut buffer_new = vec![0u8; StochasticInput::<u64>::SIZE];
        let mut buffer_tuple = vec![0u8; StochasticInput::<u64>::SIZE];

        let encoded_new = unsafe { from_new.encode_with_buffer(&mut buffer_new) };
        let encoded_tuple = unsafe { from_tuple.encode_with_buffer(&mut buffer_tuple) };

        assert_eq!(encoded_new, encoded_tuple);
    }
}
