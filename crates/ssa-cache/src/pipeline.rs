use std::marker::PhantomData;

use crate::{CacheStore, CanonicalEncode, Compute, Result, cache::codec::CodecEngine};

/// Two-stage pipeline node.
///
/// A [`Pipeline`] composes:
///
/// - an upstream `source` stage that produces an intermediate `M` from input `I`, and
/// - a local `transform` stage that maps `M -> O`.
///
/// The pipeline caches the *transform output* (`O`) using the canonical encoding of the original
/// input `I` as the key.
///
/// Typical usage is to build a pipeline once and execute batches via [`Compute::execute_many`].
///
/// # Caching / keyspace contract
///
/// The `cache` passed to [`Pipeline::new`] is dedicated to this pipeline node's transform stage.
/// Reusing the same underlying keyspace for different compute logic is not supported and may
/// cause cross-node collisions for identical input bytes.
///
/// Keyspace compatibility is caller-managed: if source/transform logic, encoding, or output types
/// change incompatibly, use a new keyspace.
pub struct Pipeline<S, C, T, I, M, O> {
    source: S,
    cache: C,
    transform: T,
    _phantom: PhantomData<(I, M, O)>,
}

impl<S, C, T, I, M, O> Pipeline<S, C, T, I, M, O> {
    /// Create a pipeline from `source`, transform `cache`, and `transform`.
    ///
    /// - `source`: upstream node that produces `M`.
    /// - `cache`: cache used for transform outputs.
    /// - `transform`: local mapping from `M` to `O`.
    ///
    /// Transform outputs are cached using the canonical encoding of the original input as the key.
    ///
    /// The `cache` argument should be dedicated to this pipeline node (see [`Pipeline`] cache
    /// keyspace contract).
    pub fn new(source: S, cache: C, transform: T) -> Self {
        Self {
            source,
            cache,
            transform,
            _phantom: PhantomData,
        }
    }
}

impl<S: Clone, C: Clone, T: Clone, I, M, O> Clone for Pipeline<S, C, T, I, M, O> {
    fn clone(&self) -> Self {
        Self {
            source: self.source.clone(),
            cache: self.cache.clone(),
            transform: self.transform.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<S, C, T, I, M, O> Compute for Pipeline<S, C, T, I, M, O>
where
    S: Compute<Input = I, Output = M>,
    C: CacheStore,
    T: Fn(M) -> Result<O>,
    I: CanonicalEncode,
    S::Engine: CodecEngine<M> + CodecEngine<O>,
{
    type Engine = S::Engine;
    type Input = I;
    type Output = O;

    fn make_engine(&self) -> Self::Engine {
        self.source.make_engine()
    }

    fn execute_with_encoded_input(
        &mut self,
        input: Self::Input,
        encoded: &[u8],
        engine: &mut Self::Engine,
    ) -> Result<Self::Output> {
        let cache = &self.cache;
        let source = &mut self.source;
        let transform = &self.transform;

        cache.fetch_or_execute::<O, Self::Engine, _>(encoded, engine, |engine_buffer| {
            let intermediate = source.execute_with_encoded_input(input, encoded, engine_buffer)?;
            transform(intermediate)
        })
    }
}

/// Extension trait for [`Compute`] to support chainable pipeline construction.
///
/// This trait provides the [`pipe`](PipelineExt::pipe) method, allowing you to wrap any
/// compute node into a [`Pipeline`] stage in a fluent, chainable manner.
///
/// # Example
///
/// ```rust
/// # use ssa_cache::prelude::*;
/// # #[cfg(feature = "bitcode")]
/// # {
/// type Store = HashMapStore<std::collections::hash_map::RandomState>;
/// let stage1 = DeterministicStep::new(Store::default(), |i: usize| Ok(i + 1), Bitcode::default);
/// let pipeline = stage1
///     .pipe(Store::default(), |i| Ok(i * 2))
///     .pipe(Store::default(), |i| Ok(format!("Result: {}", i)));
/// # }
/// ```
pub trait PipelineExt: Compute + Sized {
    /// Chain a new transform stage onto this compute node.
    ///
    /// This creates a [`Pipeline`] where `self` is the upstream source and the provided
    /// `transform` function is the local processing stage.
    ///
    /// The `cache` is used to store the outputs of this specific transform stage, keyed by the
    /// canonical encoding of the original input to the upstream source.
    fn pipe<C, T, O>(
        self,
        cache: C,
        transform: T,
    ) -> Pipeline<Self, C, T, Self::Input, Self::Output, O>
    where
        T: Fn(Self::Output) -> Result<O>,
        C: CacheStore,
        Self::Engine: CodecEngine<Self::Output> + CodecEngine<O>,
    {
        Pipeline::new(self, cache, transform)
    }
}

impl<T: Compute> PipelineExt for T {}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use std::{
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
        thread::sleep,
        time::Duration,
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
    fn test_pipeline_two_stage_caching() -> Result<()> {
        let stage1_calls = Arc::new(AtomicUsize::new(0));
        let stage1_calls_clone = stage1_calls.clone();
        let stage2_calls = Arc::new(AtomicUsize::new(0));
        let stage2_calls_clone = stage2_calls.clone();

        let pipeline = DeterministicStep::new(
            DefaultHashMapStore::default(),
            move |input| {
                stage1_calls_clone.fetch_add(1, Ordering::SeqCst);
                sleep(Duration::from_millis(10));
                Ok(input * 2)
            },
            FixtureEngine::default,
        )
        .pipe(DefaultHashMapStore::default(), move |intermediate| {
            stage2_calls_clone.fetch_add(1, Ordering::SeqCst);
            sleep(Duration::from_millis(10));
            Ok(intermediate + 10)
        });

        let results1 = pipeline
            .execute_many((0..5usize).into_par_iter(), ExecuteOptions::default())?
            .collect::<Result<Vec<usize>>>()?;

        let expected: Vec<usize> = (0..5).map(|i| i * 2 + 10).collect();
        assert_eq!(results1, expected);
        assert_eq!(stage1_calls.load(Ordering::SeqCst), 5);
        assert_eq!(stage2_calls.load(Ordering::SeqCst), 5);

        let results2 = pipeline
            .execute_many((0..5).into_par_iter(), ExecuteOptions::default())?
            .collect::<Result<Vec<usize>>>()?;

        assert_eq!(results2, expected);
        assert_eq!(stage1_calls.load(Ordering::SeqCst), 5);
        assert_eq!(stage2_calls.load(Ordering::SeqCst), 5);

        Ok(())
    }

    #[test]
    fn test_pipeline_source_and_transform_cache_split() -> Result<()> {
        let stage1_calls = Arc::new(AtomicUsize::new(0));
        let stage2_calls = Arc::new(AtomicUsize::new(0));
        let stage1_cache = DefaultHashMapStore::default();

        let stage1 = {
            let stage1_calls = stage1_calls.clone();
            DeterministicStep::new(
                stage1_cache.clone(),
                move |input| {
                    stage1_calls.fetch_add(1, Ordering::SeqCst);
                    sleep(Duration::from_millis(10));
                    Ok(input * 2)
                },
                FixtureEngine::default,
            )
        };

        let pipeline1 = {
            let stage2_calls = stage2_calls.clone();
            stage1.pipe(DefaultHashMapStore::default(), move |intermediate| {
                stage2_calls.fetch_add(1, Ordering::SeqCst);
                sleep(Duration::from_millis(10));
                Ok(format!("Result: {}", intermediate))
            })
        };

        let results1 = pipeline1
            .execute_many((0..3).into_par_iter(), ExecuteOptions::default())?
            .collect::<Result<Vec<String>>>()?;

        let expected: Vec<String> = (0..3).map(|i| format!("Result: {}", i * 2)).collect();
        assert_eq!(results1, expected);
        assert_eq!(stage1_calls.load(Ordering::SeqCst), 3);
        assert_eq!(stage2_calls.load(Ordering::SeqCst), 3);

        let stage1_reused = {
            let stage1_calls = stage1_calls.clone();
            DeterministicStep::new(
                stage1_cache,
                move |input| {
                    stage1_calls.fetch_add(1, Ordering::SeqCst);
                    sleep(Duration::from_millis(10));
                    Ok(input * 2)
                },
                FixtureEngine::default,
            )
        };
        let pipeline2 = {
            let stage2_calls = stage2_calls.clone();
            stage1_reused.pipe(DefaultHashMapStore::default(), move |intermediate| {
                stage2_calls.fetch_add(1, Ordering::SeqCst);
                sleep(Duration::from_millis(10));
                Ok(format!("Result: {}", intermediate))
            })
        };

        let results2 = pipeline2
            .execute_many((0..3).into_par_iter(), ExecuteOptions::default())?
            .collect::<Result<Vec<String>>>()?;

        assert_eq!(results2, expected);
        assert_eq!(stage1_calls.load(Ordering::SeqCst), 3);
        assert_eq!(stage2_calls.load(Ordering::SeqCst), 6);

        Ok(())
    }

    #[test]
    fn test_pipeline_with_different_types() -> Result<()> {
        let pipeline = DeterministicStep::new(
            DefaultHashMapStore::default(),
            |input: u32| Ok(input as u64 * 100),
            FixtureEngine::default,
        )
        .pipe(DefaultHashMapStore::default(), |intermediate: u64| {
            Ok(format!("Value: {}", intermediate))
        });

        let results = pipeline
            .execute_many((0..5u32).into_par_iter(), ExecuteOptions::default())?
            .collect::<Result<Vec<String>>>()?;

        let expected: Vec<String> = (0..5u32).map(|i| format!("Value: {}", i * 100)).collect();
        assert_eq!(results, expected);

        Ok(())
    }

    #[test]
    fn test_pipeline_error_propagation() -> Result<()> {
        let mut pipeline = DeterministicStep::new(
            DefaultHashMapStore::default(),
            |input| {
                if input == 3 {
                    Err(crate::error::Error::Interrupted)
                } else {
                    Ok(input * 2)
                }
            },
            FixtureEngine::default,
        )
        .pipe(DefaultHashMapStore::default(), |intermediate| {
            Ok(intermediate + 10)
        });

        let result = execute_one(&mut pipeline, 2usize)?;
        assert_eq!(result, 14);

        let result = execute_one(&mut pipeline, 3);
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_pipeline_stage2_error_propagation() -> Result<()> {
        let mut pipeline = DeterministicStep::new(
            DefaultHashMapStore::default(),
            |input| Ok(input * 2),
            FixtureEngine::default,
        )
        .pipe(DefaultHashMapStore::default(), |intermediate| {
            if intermediate == 6 {
                Err(crate::error::Error::Interrupted)
            } else {
                Ok(intermediate + 10)
            }
        });

        let result = execute_one(&mut pipeline, 2usize)?;
        assert_eq!(result, 14);

        let result = execute_one(&mut pipeline, 3);
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_pipeline_stochastic_monte_carlo_analysis() -> Result<()> {
        let experiment_calls = Arc::new(AtomicUsize::new(0));
        let experiment_calls_clone = experiment_calls.clone();
        let analysis_calls = Arc::new(AtomicUsize::new(0));
        let analysis_calls_clone = analysis_calls.clone();

        let pipeline = StochasticStep::new(
            DefaultHashMapStore::default(),
            b"mc-pi-experiment",
            move |rng, n_samples| {
                experiment_calls_clone.fetch_add(1, Ordering::SeqCst);
                let mut inside = 0usize;

                for _ in 0..n_samples {
                    let x = (rng.next_u64() as f64) / (u64::MAX as f64);
                    let y = (rng.next_u64() as f64) / (u64::MAX as f64);
                    if x * x + y * y <= 1.0 {
                        inside += 1;
                    }
                }

                Ok(4.0 * (inside as f64) / (n_samples as f64))
            },
            FixtureEngine::default,
        )
        .pipe(DefaultHashMapStore::default(), move |pi_estimate: f64| {
            analysis_calls_clone.fetch_add(1, Ordering::SeqCst);
            Ok((pi_estimate - std::f64::consts::PI).abs())
        });

        let inputs: Vec<_> = (0..8u64)
            .map(|repetition| StochasticInput::new(5_000usize, repetition))
            .collect();

        let results1 = pipeline
            .execute_many(inputs.clone().into_par_iter(), ExecuteOptions::default())?
            .collect::<Result<Vec<f64>>>()?;

        assert_eq!(results1.len(), inputs.len());
        assert!(results1.iter().all(|error| error.is_finite()));
        assert_eq!(experiment_calls.load(Ordering::SeqCst), inputs.len());
        assert_eq!(analysis_calls.load(Ordering::SeqCst), inputs.len());

        let results2 = pipeline
            .execute_many(inputs.into_par_iter(), ExecuteOptions::default())?
            .collect::<Result<Vec<f64>>>()?;

        assert_eq!(results1, results2);
        assert_eq!(experiment_calls.load(Ordering::SeqCst), 8);
        assert_eq!(analysis_calls.load(Ordering::SeqCst), 8);

        Ok(())
    }
}
