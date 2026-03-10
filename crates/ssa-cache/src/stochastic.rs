use std::marker::PhantomData;

use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::{
    CacheStore, CanonicalEncode, Compute, Result,
    cache::{
        codec::{CodecEngine, EngineFactory},
        storage::WorkerForkStore,
    },
};

const KEY_DOMAIN: &str = "ssa-cache/stochastic/key-material/v1";

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

/// Stochastic compute node with reproducible randomness.
///
/// `StochasticStep` is a cached compute node where each input is executed with a deterministic RNG
/// stream derived from `(key_material, encoded_input_bytes)`.
///
/// # When to use
///
/// - You want repeated stochastic simulations with reproducible results per repetition.
/// - You want randomness to be input-scoped (no global RNG state).
/// - You want caching keyed by canonical input bytes (including `repetition_index`).
///
/// # Reproducibility
///
/// The RNG stream for an input is derived from:
///
/// - `key_material` (provided at construction), and
/// - the canonical encoding of the full input (`param` plus `repetition_index`).
///
/// Changing either will change the stream.
///
/// # Example
///
/// ```rust
/// # use ssa_cache::prelude::*;
/// # use rayon::prelude::*;
/// # use rand::Rng;
/// # #[cfg(feature = "bitcode")]
/// # fn main() -> ssa_cache::error::Result<()> {
/// type Store = HashMapStore<std::collections::hash_map::RandomState>;
/// let step = StochasticStep::new(
///     Store::default(),
///     "my-experiment-seed",
///     |rng, param: f64| {
///         let noise: f64 = rng.random_range(-1.0..1.0);
///         Ok(param + noise)
///     },
///     Bitcode::default,
/// );
///
/// // Create 5 repetitions for param 10.0
/// let inputs = (0..5)
///     .into_par_iter()
///     .map(|idx| StochasticInput::new(10.0, idx));
/// let results = step
///     .execute_many(inputs, ExecuteOptions::default())?
///     .collect::<ssa_cache::error::Result<Vec<f64>>>()?;
/// # Ok(())
/// # }
/// # #[cfg(not(feature = "bitcode"))]
/// # fn main() {}
/// ```
///
/// # Caching / keyspace contract
///
/// The provided `cache` is treated as this step's private keyspace.
/// Reusing the same underlying keyspace for different stochastic steps is not supported and may
/// return cached values written by another step for the same input bytes.
/// Keyspace compatibility is caller-managed: when seed derivation, stochastic algorithm, or
/// encoding semantics change in an incompatible way, use a new keyspace.
///
/// Note: `key_material` only affects random seed derivation. It is not used as a cache-key
/// namespace.
#[derive(Debug)]
pub struct StochasticStep<C, P, O, F, EF> {
    cache: C,
    seed_key: [u8; 32],
    function: F,
    engine_factory: EF,
    _phantom: PhantomData<(P, O)>,
}

impl<C, P, O, F, EF> StochasticStep<C, P, O, F, EF>
where
    F: Fn(&mut Xoshiro256PlusPlus, P) -> Result<O>,
    EF: EngineFactory,
    EF::Engine: CodecEngine<O>,
{
    /// Create a stochastic step from `cache`, `key_material`, and `function`.
    ///
    /// `cache` should be dedicated to this step (see [`StochasticStep`] cache keyspace contract).
    /// `key_material` affects seed derivation only; it does not namespace cache keys.
    pub fn new(cache: C, key_material: impl AsRef<[u8]>, function: F, engine_factory: EF) -> Self {
        Self {
            cache,
            seed_key: blake3::derive_key(KEY_DOMAIN, key_material.as_ref()),
            function,
            engine_factory,
            _phantom: PhantomData,
        }
    }
}

impl<C, P, O, F, EF> StochasticStep<C, P, O, F, EF>
where
    F: Fn(&mut Xoshiro256PlusPlus, P) -> Result<O>,
{
    fn derive_seed(&self, encoded_input: &[u8]) -> [u8; 32] {
        *blake3::keyed_hash(&self.seed_key, encoded_input).as_bytes()
    }
}

impl<C: WorkerForkStore, P, O, F: Clone, EF: Clone> Clone for StochasticStep<C, P, O, F, EF> {
    fn clone(&self) -> Self {
        Self {
            cache: self.cache.fork_store(),
            seed_key: self.seed_key,
            function: self.function.clone(),
            engine_factory: self.engine_factory.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<C, P, O, F, EF> Compute for StochasticStep<C, P, O, F, EF>
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
        let seed = self.derive_seed(encoded);
        let param = input.param;
        let cache = &self.cache;
        let function = &self.function;

        cache.fetch_or_execute(encoded, engine, |_| {
            let mut rng = Xoshiro256PlusPlus::from_seed(seed);
            function(&mut rng, param)
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

    use rand::RngCore;
    use rayon::prelude::*;

    use super::*;
    use crate::{
        cache::{codec::fixtures::FixtureEngine, storage::DefaultHashMapStore},
        prelude::*,
        test_utils::execute_one,
    };

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
