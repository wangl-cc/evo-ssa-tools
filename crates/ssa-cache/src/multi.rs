use std::marker::PhantomData;

use rand::{Rng, SeedableRng};

use crate::{CacheStore, CodecEngine, Compute, Decode, Encode, Result};

pub struct ExpAnalysis<I, M, O, E, A, G, C> {
    experiment: E,
    analysis: A,
    rng: G,
    _phantom: PhantomData<(I, M, O, C)>,
}

impl<I, M, O, E, A, G, C> ExpAnalysis<I, M, O, E, A, G, C>
where
    E: Fn(&mut G, I) -> Result<M>,
    A: Fn(M) -> Result<O>,
    G: Rng + SeedableRng,
{
    pub fn new(experiment: E, analysis: A) -> Self {
        Self {
            experiment,
            analysis,
            rng: G::from_rng(&mut rand::rng()),
            _phantom: PhantomData,
        }
    }

    /// Reseed the internal RNG with a new seed
    pub fn reseed(&mut self, seed: G::Seed) {
        self.rng = G::from_seed(seed);
    }
}

impl<I, M, O, E: Clone, A: Clone, G: SeedableRng, C> Clone for ExpAnalysis<I, M, O, E, A, G, C> {
    fn clone(&self) -> Self {
        Self {
            experiment: self.experiment.clone(),
            analysis: self.analysis.clone(),
            rng: G::from_rng(&mut rand::rng()),
            _phantom: PhantomData,
        }
    }
}

impl<I, M, O, E, A, G, C1, C2, C> Compute<(C1, C2)> for ExpAnalysis<I, M, O, E, A, G, C>
where
    C1: CacheStore<C>,
    C2: CacheStore<C>,
    E: Fn(&mut G, I) -> Result<M>,
    A: Fn(M) -> Result<O>,
    C: CodecEngine + Encode<I> + Encode<usize> + Encode<M> + Decode<M> + Encode<O> + Decode<O>,
{
    type Engine = C;
    type Input = (usize, I);
    type Output = O;

    fn execute(
        &mut self,
        input: Self::Input,
        cache: &(C1, C2),
        engine: &mut Self::Engine,
    ) -> Result<Self::Output> {
        let (i, input) = input;
        let mut sig = engine.encode(&input).to_vec();
        let index = engine.encode(&i).to_vec();
        sig.extend_from_slice(&index);
        let sig = sig.as_ref();

        // Check if the final output is already cached
        if let Some(output) = cache.1.fetch(sig, engine)? {
            Ok(output)
        } else {
            // Check if the first output is already cached
            let inter = if let Some(output) = cache.0.fetch(sig, engine)? {
                output
            } else {
                let inter = (self.experiment)(&mut self.rng, input)?;
                cache.0.store(sig, engine, &inter)?;
                inter
            };
            let output = (self.analysis)(inter)?;
            cache.1.store(sig, engine, &output)?;
            Ok(output)
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use std::{
        hash::RandomState,
        sync::{
            Arc,
            atomic::{AtomicBool, AtomicUsize, Ordering},
        },
        thread::sleep,
        time::Duration,
    };

    use rand::rngs::SmallRng;
    use rayon::prelude::*;

    use super::*;
    use crate::HashMapStore;

    type Engine = crate::BitcodeCodec;

    #[test]
    fn test_basic() -> Result<()> {
        let exp_analysis: ExpAnalysis<usize, usize, usize, _, _, SmallRng, Engine> =
            ExpAnalysis::new(
                |_: &mut SmallRng, input| Ok(input * 2), // experiment: double the input
                |intermediate| Ok(intermediate + 10),    // analysis: add 10 to intermediate result
            );

        let cache1 = HashMapStore::<RandomState>::default();
        let cache2 = HashMapStore::<RandomState>::default();
        let cache = (cache1, cache2);

        let results = exp_analysis
            .execute_many(
                &cache,
                Arc::new(AtomicBool::new(false)),
                (0..10).into_par_iter().map(|i| (i, i)),
            )?
            .collect::<Result<Vec<usize>>>()?;

        let expected: Vec<usize> = (0..10).map(|i| i * 2 + 10).collect();
        assert_eq!(results, expected);

        Ok(())
    }

    #[test]
    fn test_caching() -> Result<()> {
        // Track how many times each function is called
        let experiment_calls = Arc::new(AtomicUsize::new(0));
        let analysis_calls = Arc::new(AtomicUsize::new(0));

        let exp_calls_clone = experiment_calls.clone();
        let analysis_calls_clone = analysis_calls.clone();

        let exp_analysis: ExpAnalysis<usize, usize, usize, _, _, SmallRng, Engine> =
            ExpAnalysis::new(
                move |_: &mut SmallRng, input| {
                    exp_calls_clone.fetch_add(1, Ordering::SeqCst);
                    sleep(Duration::from_millis(10)); // Simulate work
                    Ok(input * 3)
                },
                move |intermediate| {
                    analysis_calls_clone.fetch_add(1, Ordering::SeqCst);
                    sleep(Duration::from_millis(10)); // Simulate work
                    Ok(intermediate + 5)
                },
            );

        let cache1 = HashMapStore::<RandomState>::default();
        let cache2 = HashMapStore::<RandomState>::default();
        let cache = (cache1, cache2);

        // First execution - both stages should run
        let results1 = exp_analysis
            .execute_many(
                &cache,
                Arc::new(AtomicBool::new(false)),
                (0..5).into_par_iter().map(|i| (i, i)),
            )?
            .collect::<Result<Vec<usize>>>()?;

        let expected: Vec<usize> = (0..5).map(|i| i * 3 + 5).collect();
        assert_eq!(results1, expected);
        assert_eq!(experiment_calls.load(Ordering::SeqCst), 5);
        assert_eq!(analysis_calls.load(Ordering::SeqCst), 5);

        // Second execution - should use cached final results
        let results2 = exp_analysis
            .execute_many(
                &cache,
                Arc::new(AtomicBool::new(false)),
                (0..5).into_par_iter().map(|i| (i, i)),
            )?
            .collect::<Result<Vec<usize>>>()?;

        assert_eq!(results2, expected);
        // No additional calls should have been made
        assert_eq!(experiment_calls.load(Ordering::SeqCst), 5);
        assert_eq!(analysis_calls.load(Ordering::SeqCst), 5);

        Ok(())
    }

    #[test]
    fn test_intermediate_caching() -> Result<()> {
        let experiment_calls = Arc::new(AtomicUsize::new(0));
        let analysis_calls = Arc::new(AtomicUsize::new(0));

        let exp_calls_clone = experiment_calls.clone();
        let analysis_calls_clone = analysis_calls.clone();

        let exp_analysis: ExpAnalysis<usize, usize, String, _, _, SmallRng, Engine> =
            ExpAnalysis::new(
                move |_: &mut SmallRng, input| {
                    exp_calls_clone.fetch_add(1, Ordering::SeqCst);
                    sleep(Duration::from_millis(10));
                    Ok(input * 2)
                },
                move |intermediate| {
                    analysis_calls_clone.fetch_add(1, Ordering::SeqCst);
                    sleep(Duration::from_millis(10));
                    Ok(format!("result_{intermediate}"))
                },
            );

        let cache1 = HashMapStore::<RandomState>::default();
        let cache2 = HashMapStore::<RandomState>::default();
        let cache = (cache1, cache2);

        // First execution
        let results1 = exp_analysis
            .execute_many(
                &cache,
                Arc::new(AtomicBool::new(false)),
                (0..3).into_par_iter().map(|i| (i, i)),
            )?
            .collect::<Result<Vec<String>>>()?;

        let expected: Vec<String> = (0..3).map(|i| format!("result_{}", i * 2)).collect();
        assert_eq!(results1, expected);
        assert_eq!(experiment_calls.load(Ordering::SeqCst), 3);
        assert_eq!(analysis_calls.load(Ordering::SeqCst), 3);

        // Clear the final results cache but keep intermediate cache
        let cache1 = cache.0; // Keep intermediate cache
        let cache2 = HashMapStore::<RandomState>::default(); // New final cache
        let cache = (cache1, cache2);

        // Second execution - should reuse intermediate results but recompute final
        let results2 = exp_analysis
            .execute_many(
                &cache,
                Arc::new(AtomicBool::new(false)),
                (0..3).into_par_iter().map(|i| (i, i)),
            )?
            .collect::<Result<Vec<String>>>()?;

        assert_eq!(results2, expected);
        // Experiment should not be called again (cached intermediate)
        assert_eq!(experiment_calls.load(Ordering::SeqCst), 3);
        // Analysis should be called again (no final cache)
        assert_eq!(analysis_calls.load(Ordering::SeqCst), 6);

        Ok(())
    }

    #[test]
    fn test_reseed() -> Result<()> {
        use rand::Rng;

        // Create ExpAnalysis that uses RNG in experiment
        let mut exp_analysis: ExpAnalysis<u32, u32, u32, _, _, SmallRng, Engine> = ExpAnalysis::new(
            |rng: &mut SmallRng, input| Ok(input + rng.random::<u32>()),
            |x| Ok(x + 1),
        );
        let seed = [42u8; 32];
        let cache = ((), ());
        let mut engine = Engine::default();

        // First run with seed
        exp_analysis.reseed(seed);
        let out1 = exp_analysis.execute((0, 5), &cache, &mut engine)?;

        // Second run with same seed (should produce same result)
        exp_analysis.reseed(seed);
        let out2 = exp_analysis.execute((0, 5), &cache, &mut engine)?;

        // Verify consistent results with same seed, different with new seed
        assert_eq!(out1, out2);

        Ok(())
    }
}
