//! Reproducible stochastic compute steps.
//!
//! This module provides [`StochasticStep`], [`StochasticInput`], and seed/domain types for
//! deterministic random stream construction.
//!
//! For simulations with multiple random subsystems, prefer configuring named domain streams up
//! front. Each domain gets an independent deterministic stream for every input:
//!
//! ```rust
//! # use ssa_pipeline::prelude::*;
//! # use rand::Rng;
//! # #[cfg(feature = "bitcode")]
//! # {
//! const DIVISION_EVENT_STREAM: StreamDomain = StreamDomain::new("cell-model/division-event/v1");
//! const COPY_NUMBER_SEGREGATION_STREAM: StreamDomain =
//!     StreamDomain::new("cell-model/copy-number-segregation/v1");
//!
//! let _step = StochasticStep::new_with_domain_streams(
//!     DefaultHashMapStore::default(),
//!     "experiment-seed",
//!     [DIVISION_EVENT_STREAM, COPY_NUMBER_SEGREGATION_STREAM],
//!     |streams, parent_copy_number: u64| {
//!         let [division_rng, segregation_rng] = streams.as_mut();
//!
//!         let divides = division_rng.random::<u64>() % 4 == 0;
//!         if !divides {
//!             return Ok((parent_copy_number, 0));
//!         }
//!
//!         let replicated = parent_copy_number * 2;
//!         let left_daughter = segregation_rng.random_range(0..=replicated);
//!         let right_daughter = replicated - left_daughter;
//!
//!         Ok((left_daughter, right_daughter))
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
//! # use rand::Rng;
//! # #[cfg(feature = "bitcode")]
//! # {
//! let _step = StochasticStep::new(
//!     DefaultHashMapStore::default(),
//!     "experiment-seed",
//!     |rng, param: u64| Ok(rng.random::<u64>() ^ param),
//!     Bitcode06::default,
//! );
//! # }
//! ```
//!
//! Seed derivation differs by mode:
//!
//! ```text
//! single stream: SeedDomain + seed_material -> RootSeed
//! single stream: RootSeed + encoded_input -> Xoshiro256PlusPlus
//!
//! domain streams: SeedDomain + seed_material -> RootSeed
//! domain streams: RootSeed + StreamDomain -> DomainSeed
//! domain streams: DomainSeed + encoded_input -> Xoshiro256PlusPlus
//! ```
//!
//! For design details, see `crates/ssa-pipeline/RANDOM_STREAMS.md`.

use std::marker::PhantomData;

use rand::rngs::Xoshiro256PlusPlus;

pub mod seed;

pub use seed::{
    DomainSeed, DomainSeeds, RootSeed, STOCHASTIC_ROOT_SEED_DOMAIN, SeedDomain, StochasticStreams,
    StreamDomain,
};

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
/// `StochasticStep` is shared by the single-stream and domain-stream constructors. The stored
/// `seed` field is generic:
///
/// - `StochasticStep<C, P, O, RootSeed, F, EF>` is created by [`StochasticStep::new`] and stores
///   one [`RootSeed`]. Each input uses `RootSeed + encoded_input -> Xoshiro256PlusPlus`.
/// - `StochasticStep<C, P, O, DomainSeeds<N>, F, EF>` is created by
///   [`StochasticStep::new_with_domain_streams`] and stores pre-derived [`DomainSeeds<N>`]. Each
///   input uses `DomainSeeds<N> + encoded_input -> StochasticStreams<N>`.
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
/// Note: `seed_material` only affects random seed derivation. It is not used as a cache-key
/// namespace.
#[derive(Debug)]
pub struct StochasticStep<C, P, O, S, F, EF> {
    cache: C,
    seed: S,
    function: F,
    engine_factory: EF,
    _phantom: PhantomData<(P, O)>,
}

impl<C, P, O, F, EF> StochasticStep<C, P, O, RootSeed, F, EF>
where
    F: Fn(&mut Xoshiro256PlusPlus, P) -> Result<O>,
    EF: EngineFactory,
    EF::Engine: CodecEngine<O>,
{
    /// Create a stochastic step with a single RNG stream.
    ///
    /// `cache` should be dedicated to this step (see [`StochasticStep`] cache keyspace contract).
    /// `seed_material` affects seed derivation only; it does not namespace cache keys.
    pub fn new(cache: C, seed_material: impl AsRef<[u8]>, function: F, engine_factory: EF) -> Self {
        Self {
            cache,
            seed: RootSeed::from_domain(STOCHASTIC_ROOT_SEED_DOMAIN, seed_material),
            function,
            engine_factory,
            _phantom: PhantomData,
        }
    }
}

impl<C, P, O, F, EF, const N: usize> StochasticStep<C, P, O, DomainSeeds<N>, F, EF>
where
    F: Fn(&mut StochasticStreams<N>, P) -> Result<O>,
    EF: EngineFactory,
    EF::Engine: CodecEngine<O>,
{
    /// Create a stochastic step whose function receives configured domain streams.
    ///
    /// `cache` should be dedicated to this step (see [`StochasticStep`] cache keyspace contract).
    /// `seed_material` affects seed derivation only; it does not namespace cache keys.
    pub fn new_with_domain_streams(
        cache: C,
        seed_material: impl AsRef<[u8]>,
        domains: [StreamDomain; N],
        function: F,
        engine_factory: EF,
    ) -> Self {
        let root_seed = RootSeed::from_domain(STOCHASTIC_ROOT_SEED_DOMAIN, seed_material);

        Self {
            cache,
            seed: root_seed.derive_domain_seeds(domains),
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

impl<C, P, O, F, EF> Compute for StochasticStep<C, P, O, RootSeed, F, EF>
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

impl<C, P, O, F, EF, const N: usize> Compute for StochasticStep<C, P, O, DomainSeeds<N>, F, EF>
where
    F: Fn(&mut StochasticStreams<N>, P) -> Result<O>,
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

    const TEST_SEED_DOMAIN: SeedDomain = SeedDomain::new("ssa-pipeline/test/root-seed/v1");
    const MAIN_STREAM: StreamDomain = StreamDomain::new("ssa/main/v1");
    const SEGREGATION_STREAM: StreamDomain = StreamDomain::new("model/segregation/v1");
    const MUTATION_STREAM: StreamDomain = StreamDomain::new("model/mutation/v1");

    #[test]
    fn test_seed_and_stream_domain_names_are_stable() {
        let seed_domain = SeedDomain::new("test/root/v1");
        let stream_domain = StreamDomain::new("test/stream/v1");

        assert_eq!(seed_domain.as_str(), "test/root/v1");
        assert_eq!(stream_domain.as_str(), "test/stream/v1");
        assert_eq!(seed_domain.to_string(), "test/root/v1");
        assert_eq!(stream_domain.to_string(), "test/stream/v1");
    }

    #[test]
    fn test_seed_debug_output_is_redacted() {
        let root_seed = RootSeed::from_domain(TEST_SEED_DOMAIN, b"root");
        let domain_seed = root_seed.derive_domain_seed(MAIN_STREAM);

        let root_debug = format!("{root_seed:?}");
        let domain_debug = format!("{domain_seed:?}");

        assert_eq!(root_debug, "RootSeed { .. }");
        assert_eq!(domain_debug, "DomainSeed { .. }");
        assert!(!root_debug.contains("bytes"));
        assert!(!domain_debug.contains("bytes"));
    }

    #[test]
    fn test_domain_seed_streams_are_stable_and_isolated() {
        let root_seed = RootSeed::from_domain(TEST_SEED_DOMAIN, b"root");
        let segregation_seed = root_seed.derive_domain_seed(SEGREGATION_STREAM);
        let mutation_seed = root_seed.derive_domain_seed(MUTATION_STREAM);
        let mut segregation1 = segregation_seed.make_stream(b"input-A");
        let mut segregation2 = segregation_seed.make_stream(b"input-A");
        let mut mutation = mutation_seed.make_stream(b"input-A");

        assert_eq!(segregation1.next_u64(), segregation2.next_u64());
        assert_ne!(segregation1.next_u64(), mutation.next_u64());
    }

    #[test]
    fn test_domain_seed_input_bytes_change_stream() {
        let root_seed = RootSeed::from_domain(TEST_SEED_DOMAIN, b"root");
        let seed = root_seed.derive_domain_seed(SEGREGATION_STREAM);
        let mut rng_a = seed.make_stream(b"input-A");
        let mut rng_b = seed.make_stream(b"input-B");

        assert_ne!(rng_a.next_u64(), rng_b.next_u64());
    }

    #[test]
    fn test_domain_seed_bundle_allows_duplicate_domains() {
        let root_seed = RootSeed::from_domain(TEST_SEED_DOMAIN, b"root");
        let seeds = root_seed.derive_domain_seeds([SEGREGATION_STREAM, SEGREGATION_STREAM]);
        let mut streams = seeds.make_streams(b"input-A");
        let [left, right] = streams.as_mut();

        assert_eq!(left.next_u64(), right.next_u64());
    }

    #[test]
    fn test_domain_seed_bundle_accessors_preserve_order() {
        let root_seed = RootSeed::from_domain(TEST_SEED_DOMAIN, b"root");
        let seeds = root_seed.derive_domain_seeds([SEGREGATION_STREAM, MUTATION_STREAM]);
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
    fn test_root_seed_make_stream_does_not_use_domain_seed() {
        let root_seed = RootSeed::from_domain(TEST_SEED_DOMAIN, b"root");
        let domain_seed = root_seed.derive_domain_seed(MAIN_STREAM);
        let mut single_rng = root_seed.make_stream(b"input-A");
        let mut domain_rng = domain_seed.make_stream(b"input-A");

        assert_ne!(single_rng.next_u64(), domain_rng.next_u64());
    }

    #[test]
    fn test_domain_seed_bundle_supports_multiple_mutable_rngs() {
        let root_seed = RootSeed::from_domain(TEST_SEED_DOMAIN, b"root");
        let seeds = root_seed.derive_domain_seeds([SEGREGATION_STREAM, MUTATION_STREAM]);
        let mut streams = seeds.make_streams(b"input-A");
        let [segregation_rng, mutation_rng] = streams.as_mut();

        let segregation_value = segregation_rng.next_u64();
        let mutation_value = mutation_rng.next_u64();

        assert_ne!(segregation_value, mutation_value);
    }

    #[test]
    fn test_stochastic_stream_bundle_into_inner_preserves_order() {
        let root_seed = RootSeed::from_domain(TEST_SEED_DOMAIN, b"root");
        let seeds = root_seed.derive_domain_seeds([SEGREGATION_STREAM, MUTATION_STREAM]);
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
    fn test_stochastic_step_reproducible_with_domain_streams() -> Result<()> {
        let mut step = StochasticStep::new_with_domain_streams(
            (),
            b"experiment-A",
            [MAIN_STREAM, SEGREGATION_STREAM],
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
            b"experiment-A",
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
            b"experiment-A",
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
    fn test_stochastic_diff_seed_material_isolated() -> Result<()> {
        let mut step1 = StochasticStep::new(
            (),
            b"experiment-A",
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
            b"experiment-B",
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
            b"experiment-A",
            move |rng, param| {
                calls_a_clone.fetch_add(1, Ordering::SeqCst);
                Ok(rng.next_u64() ^ param)
            },
            FixtureEngine::default,
        );
        let mut step_b = StochasticStep::new(
            DefaultHashMapStore::default(),
            b"experiment-B",
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
            b"experiment-A",
            |rng, param| Ok(rng.next_u64() ^ param),
            FixtureEngine::default,
        );
        let step2 = StochasticStep::new(
            DefaultHashMapStore::default(),
            b"experiment-A",
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
            b"experiment-A",
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
