use std::marker::PhantomData;

use rand::{Rng, SeedableRng};

use crate::{CacheStore, Cacheable, Compute, Encodeable, Result};

pub struct ExpAnalysis<I, M, O, E, A, G> {
    experiment: E,
    analysis: A,
    rng: G,
    _phantom: PhantomData<(I, M, O)>,
}

impl<I, M, O, E, A, G> ExpAnalysis<I, M, O, E, A, G>
where
    E: Fn(&mut G, I) -> M,
    A: Fn(M) -> O,
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

impl<I, M, O, E: Clone, A: Clone, G: SeedableRng> Clone for ExpAnalysis<I, M, O, E, A, G> {
    fn clone(&self) -> Self {
        Self {
            experiment: self.experiment.clone(),
            analysis: self.analysis.clone(),
            rng: G::from_rng(&mut rand::rng()),
            _phantom: PhantomData,
        }
    }
}

impl<I, M, O, E, A, G, C1, C2> Compute<(C1, C2)> for ExpAnalysis<I, M, O, E, A, G>
where
    C1: CacheStore,
    C2: CacheStore,
    I: Encodeable,
    M: Cacheable<Buffer = I::Buffer>,
    O: Cacheable<Buffer = I::Buffer>,
    E: Fn(&mut G, I) -> M,
    A: Fn(M) -> O,
{
    type Input = (usize, I);
    type Output = O;

    fn execute(
        &mut self,
        input: Self::Input,
        cache: &(C1, C2),
        buffer: &mut <Self::Output as Encodeable>::Buffer,
    ) -> Result<Self::Output> {
        let (i, input) = input;
        let sig = [input.encode(buffer), &i.to_le_bytes()].concat();
        let sig = sig.as_ref();

        // Check if the final output is already cached
        if let Some(output) = cache.1.fetch(sig, buffer)? {
            Ok(output)
        } else {
            // Check if the first output is already cached
            let inter = if let Some(output) = cache.0.fetch(sig, buffer)? {
                output
            } else {
                let inter = (self.experiment)(&mut self.rng, input);
                cache.0.store(sig, buffer, &inter)?;
                inter
            };
            let output = (self.analysis)(inter);
            cache.1.store(sig, buffer, &output)?;
            Ok(output)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        hash::RandomState,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
        thread::sleep,
        time::Duration,
    };

    use rand::rngs::SmallRng;
    use rayon::prelude::*;

    use super::*;
    use crate::HashMapStore;

    #[test]
    fn test_basic() -> Result<()> {
        let exp_analysis = ExpAnalysis::new(
            |_: &mut SmallRng, input| input * 2, // experiment: double the input
            |intermediate| intermediate + 10,    // analysis: add 10 to intermediate result
        );

        let cache1 = HashMapStore::<RandomState>::default();
        let cache2 = HashMapStore::<RandomState>::default();
        let cache = (cache1, cache2);

        let results = exp_analysis
            .execute_many(&cache, (0..10).into_par_iter().map(|i| (i, i)))?
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

        let exp_analysis = ExpAnalysis::new(
            move |_: &mut SmallRng, input| {
                exp_calls_clone.fetch_add(1, Ordering::SeqCst);
                sleep(Duration::from_millis(10)); // Simulate work
                input * 3
            },
            move |intermediate| {
                analysis_calls_clone.fetch_add(1, Ordering::SeqCst);
                sleep(Duration::from_millis(10)); // Simulate work
                intermediate + 5
            },
        );

        let cache1 = HashMapStore::<RandomState>::default();
        let cache2 = HashMapStore::<RandomState>::default();
        let cache = (cache1, cache2);

        // First execution - both stages should run
        let results1 = exp_analysis
            .execute_many(&cache, (0..5).into_par_iter().map(|i| (i, i)))?
            .collect::<Result<Vec<usize>>>()?;

        let expected: Vec<usize> = (0..5).map(|i| i * 3 + 5).collect();
        assert_eq!(results1, expected);
        assert_eq!(experiment_calls.load(Ordering::SeqCst), 5);
        assert_eq!(analysis_calls.load(Ordering::SeqCst), 5);

        // Second execution - should use cached final results
        let results2 = exp_analysis
            .execute_many(&cache, (0..5).into_par_iter().map(|i| (i, i)))?
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

        let exp_analysis = ExpAnalysis::new(
            move |_: &mut SmallRng, input| {
                exp_calls_clone.fetch_add(1, Ordering::SeqCst);
                sleep(Duration::from_millis(10));
                input * 2
            },
            move |intermediate| {
                analysis_calls_clone.fetch_add(1, Ordering::SeqCst);
                sleep(Duration::from_millis(10));
                format!("result_{intermediate}")
            },
        );

        let cache1 = HashMapStore::<RandomState>::default();
        let cache2 = HashMapStore::<RandomState>::default();
        let cache = (cache1, cache2);

        // First execution
        let results1 = exp_analysis
            .execute_many(&cache, (0..3).into_par_iter().map(|i| (i, i)))?
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
            .execute_many(&cache, (0..3).into_par_iter().map(|i| (i, i)))?
            .collect::<Result<Vec<String>>>()?;

        assert_eq!(results2, expected);
        // Experiment should not be called again (cached intermediate)
        assert_eq!(experiment_calls.load(Ordering::SeqCst), 3);
        // Analysis should be called again (no final cache)
        assert_eq!(analysis_calls.load(Ordering::SeqCst), 6);

        Ok(())
    }
}
