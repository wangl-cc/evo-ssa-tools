use std::marker::PhantomData;

use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::{CacheStore, CanonicalEncode, Codec, Compute, Result};

/// Input for stochastic computation.
///
/// `param` is the model input, and `repetition_index` selects a deterministic random
/// stream for repeated runs with the same `param`.
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

/// Stochastic cached computation with deterministic RNG seeding.
///
/// Reproducibility is determined by:
///
/// - Key material passed to [`Self::new`],
/// - Canonical input bytes,
/// - The fixed RNG type (`Xoshiro256PlusPlus`),
/// - and key-material normalization domain version in this module.
///
/// Updating the key-material domain version or RNG algorithm changes both generated random streams
/// and cache signatures by design.
#[derive(Debug)]
pub struct StochasticStep<P, O, F> {
    seed_key: [u8; 32],
    function: F,
    _phantom: PhantomData<(P, O)>,
}

impl<P, O, F> StochasticStep<P, O, F>
where
    F: Fn(&mut Xoshiro256PlusPlus, P) -> Result<O>,
{
    /// Domain for key-material normalization.
    ///
    /// `key_material` is normalized to a 32-byte key with `blake3::derive_key`.
    const KEY_DOMAIN: &str = "ssa-cache/stochastic/key-material/v1";

    /// Create a stochastic step with deterministic random stream derivation.
    ///
    /// Any bytes can be used as key material (for example, experiment name bytes). The material is
    /// normalized once to a 32-byte key and then used to derive per-input run seeds.
    pub fn new(key_material: impl AsRef<[u8]>, function: F) -> Self {
        Self {
            seed_key: blake3::derive_key(Self::KEY_DOMAIN, key_material.as_ref()),
            function,
            _phantom: PhantomData,
        }
    }

    fn derive_seed(&self, input_signature: &[u8]) -> [u8; 32] {
        *blake3::keyed_hash(&self.seed_key, input_signature).as_bytes()
    }
}

impl<P, O, F: Clone> Clone for StochasticStep<P, O, F> {
    fn clone(&self) -> Self {
        Self {
            seed_key: self.seed_key,
            function: self.function.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<P, O, F, C> Compute<C> for StochasticStep<P, O, F>
where
    F: Fn(&mut Xoshiro256PlusPlus, P) -> Result<O>,
    C: CacheStore,
    P: CanonicalEncode,
    O: Codec,
{
    type Input = StochasticInput<P>;
    type Output = O;

    fn execute_with_sig(
        &mut self,
        input: Self::Input,
        input_signature: &[u8],
        cache: &C,
        codec_buffer: &mut <Self::Output as Codec>::Buffer,
    ) -> Result<Self::Output> {
        let seed = self.derive_seed(input_signature);
        let param = input.param;
        cache.fetch_or_execute(&seed, codec_buffer, |_| {
            let mut rng = Xoshiro256PlusPlus::from_seed(seed);
            (self.function)(&mut rng, param)
        })
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use std::{
        collections::hash_map::RandomState,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
    };

    use rand::RngCore;
    use rayon::prelude::*;

    use super::*;
    use crate::{CodecBuffer, ExecuteOptions, HashMapStore, Pipeline};

    #[test]
    fn test_stochastic_reproducible_same_seed_and_input() -> Result<()> {
        let mut step: StochasticStep<u64, [u64; 4], _> =
            StochasticStep::new(b"experiment-A", |rng, param| {
                Ok([
                    rng.next_u64() ^ param,
                    rng.next_u64(),
                    rng.next_u64(),
                    rng.next_u64(),
                ])
            });
        let cache = ();
        let mut encode_buffer = vec![0u8; <StochasticInput<u64> as CanonicalEncode>::SIZE];
        let mut codec_buffer = <[u64; 4] as Codec>::Buffer::init();

        let output1 = unsafe {
            step.execute(
                StochasticInput::new(42, 7),
                &cache,
                &mut encode_buffer,
                &mut codec_buffer,
            )?
        };
        let output2 = unsafe {
            step.execute(
                StochasticInput::new(42, 7),
                &cache,
                &mut encode_buffer,
                &mut codec_buffer,
            )?
        };

        assert_eq!(output1, output2);
        Ok(())
    }

    #[test]
    fn test_stochastic_diff_repetition_index_changes_stream() -> Result<()> {
        let mut step: StochasticStep<u64, [u64; 4], _> =
            StochasticStep::new(b"experiment-A", |rng, param| {
                Ok([
                    rng.next_u64() ^ param,
                    rng.next_u64(),
                    rng.next_u64(),
                    rng.next_u64(),
                ])
            });
        let cache = ();
        let mut encode_buffer = vec![0u8; <StochasticInput<u64> as CanonicalEncode>::SIZE];
        let mut codec_buffer = <[u64; 4] as Codec>::Buffer::init();

        let output1 = unsafe {
            step.execute(
                StochasticInput::new(42, 1),
                &cache,
                &mut encode_buffer,
                &mut codec_buffer,
            )?
        };
        let output2 = unsafe {
            step.execute(
                StochasticInput::new(42, 2),
                &cache,
                &mut encode_buffer,
                &mut codec_buffer,
            )?
        };

        assert_ne!(output1, output2);
        Ok(())
    }

    #[test]
    fn test_stochastic_diff_seed_material_isolated() -> Result<()> {
        let mut step1: StochasticStep<u64, [u64; 4], _> =
            StochasticStep::new(b"experiment-A", |rng, param| {
                Ok([
                    rng.next_u64() ^ param,
                    rng.next_u64(),
                    rng.next_u64(),
                    rng.next_u64(),
                ])
            });
        let mut step2: StochasticStep<u64, [u64; 4], _> =
            StochasticStep::new(b"experiment-B", |rng, param| {
                Ok([
                    rng.next_u64() ^ param,
                    rng.next_u64(),
                    rng.next_u64(),
                    rng.next_u64(),
                ])
            });
        let cache = ();
        let mut encode_buffer1 = vec![0u8; <StochasticInput<u64> as CanonicalEncode>::SIZE];
        let mut encode_buffer2 = vec![0u8; <StochasticInput<u64> as CanonicalEncode>::SIZE];
        let mut codec_buffer1 = <[u64; 4] as Codec>::Buffer::init();
        let mut codec_buffer2 = <[u64; 4] as Codec>::Buffer::init();

        let output1 = unsafe {
            step1.execute(
                StochasticInput::new(42, 7),
                &cache,
                &mut encode_buffer1,
                &mut codec_buffer1,
            )?
        };
        let output2 = unsafe {
            step2.execute(
                StochasticInput::new(42, 7),
                &cache,
                &mut encode_buffer2,
                &mut codec_buffer2,
            )?
        };

        assert_ne!(output1, output2);
        Ok(())
    }

    #[test]
    fn test_stochastic_cache_isolated_by_seed() -> Result<()> {
        let calls_a = Arc::new(AtomicUsize::new(0));
        let calls_a_clone = calls_a.clone();
        let calls_b = Arc::new(AtomicUsize::new(0));
        let calls_b_clone = calls_b.clone();

        let mut step_a: StochasticStep<u64, u64, _> =
            StochasticStep::new(b"experiment-A", move |rng, param| {
                calls_a_clone.fetch_add(1, Ordering::SeqCst);
                Ok(rng.next_u64() ^ param)
            });
        let mut step_b: StochasticStep<u64, u64, _> =
            StochasticStep::new(b"experiment-B", move |rng, param| {
                calls_b_clone.fetch_add(1, Ordering::SeqCst);
                Ok(rng.next_u64() ^ param)
            });

        let cache = HashMapStore::<RandomState>::default();
        let mut encode_buffer_a = vec![0u8; <StochasticInput<u64> as CanonicalEncode>::SIZE];
        let mut encode_buffer_b = vec![0u8; <StochasticInput<u64> as CanonicalEncode>::SIZE];
        let mut codec_buffer_a = <u64 as Codec>::Buffer::init();
        let mut codec_buffer_b = <u64 as Codec>::Buffer::init();

        let input = StochasticInput::new(42, 7);

        let output_a1 = unsafe {
            step_a.execute(
                input.clone(),
                &cache,
                &mut encode_buffer_a,
                &mut codec_buffer_a,
            )?
        };
        let output_b1 = unsafe {
            step_b.execute(
                input.clone(),
                &cache,
                &mut encode_buffer_b,
                &mut codec_buffer_b,
            )?
        };
        let output_a2 = unsafe {
            step_a.execute(
                input.clone(),
                &cache,
                &mut encode_buffer_a,
                &mut codec_buffer_a,
            )?
        };
        let output_b2 =
            unsafe { step_b.execute(input, &cache, &mut encode_buffer_b, &mut codec_buffer_b)? };

        assert_eq!(output_a1, output_a2);
        assert_eq!(output_b1, output_b2);
        assert_eq!(calls_a.load(Ordering::SeqCst), 1);
        assert_eq!(calls_b.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[test]
    fn test_stochastic_execute_many_parallel_reproducible() -> Result<()> {
        let step: StochasticStep<u64, u64, _> =
            StochasticStep::new(b"experiment-A", |rng, param| Ok(rng.next_u64() ^ param));

        let inputs: Vec<_> = (0..128u64)
            .map(|i| StochasticInput::new(i, i % 8))
            .collect();

        let cache1 = HashMapStore::<RandomState>::default();
        let cache2 = HashMapStore::<RandomState>::default();

        let outputs1 = step
            .execute_many(
                inputs.clone().into_par_iter(),
                &cache1,
                ExecuteOptions::default(),
            )?
            .collect::<Result<Vec<u64>>>()?;
        let outputs2 = step
            .execute_many(inputs.into_par_iter(), &cache2, ExecuteOptions::default())?
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

        let stage1: StochasticStep<usize, usize, _> =
            StochasticStep::new(b"experiment-A", move |rng, param| {
                stage1_calls_clone.fetch_add(1, Ordering::SeqCst);
                Ok((rng.next_u64() as usize) ^ param)
            });

        let pipeline: Pipeline<_, _, StochasticInput<usize>, usize, usize> =
            Pipeline::new(stage1, move |intermediate| {
                stage2_calls_clone.fetch_add(1, Ordering::SeqCst);
                Ok(intermediate + 10)
            });

        let cache1 = HashMapStore::<RandomState>::default();
        let cache2 = HashMapStore::<RandomState>::default();
        let cache = (cache1, cache2);
        let inputs: Vec<_> = (0..20usize)
            .map(|i| StochasticInput::new(i, (i % 4) as u64))
            .collect();

        let outputs1 = pipeline
            .execute_many(
                inputs.clone().into_par_iter(),
                &cache,
                ExecuteOptions::default(),
            )?
            .collect::<Result<Vec<usize>>>()?;
        let outputs2 = pipeline
            .execute_many(inputs.into_par_iter(), &cache, ExecuteOptions::default())?
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
