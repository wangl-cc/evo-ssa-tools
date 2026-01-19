use std::marker::PhantomData;

use rand::{Rng, SeedableRng};

use crate::{CacheStore, CodecEngine, Compute, Decode, Encode, Result, fetch_or_execute};

#[derive(Debug)]
pub struct PureCompute<I, O, F, E> {
    function: F,
    _phantom: PhantomData<(I, O, E)>,
}

impl<I, O, F, E> PureCompute<I, O, F, E>
where
    F: Fn(I) -> Result<O>,
{
    /// Create a new pure (deterministic) cached computation
    ///
    /// The function can return a `Result` to indicate computation failure.
    pub fn new(function: F) -> Self {
        Self {
            function,
            _phantom: PhantomData,
        }
    }
}

impl<I, O, F: Clone, E> Clone for PureCompute<I, O, F, E> {
    fn clone(&self) -> Self {
        Self {
            function: self.function.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<I, O, F, C, E> Compute<C> for PureCompute<I, O, F, E>
where
    F: Fn(I) -> Result<O>,
    C: CacheStore<E>,
    E: CodecEngine + Encode<I> + Encode<O> + Decode<O>,
{
    type Engine = E;
    type Input = I;
    type Output = O;

    fn execute(
        &mut self,
        input: Self::Input,
        cache: &C,
        engine: &mut Self::Engine,
    ) -> Result<Self::Output> {
        let sig = engine.encode(&input).to_vec();
        fetch_or_execute(&sig, cache, engine, || (self.function)(input))
    }
}

#[derive(Debug)]
pub struct StochasiticCompute<I, O, F, G, E> {
    function: F,
    rng: G,
    _phantom: PhantomData<(I, O, E)>,
}

impl<I, O, F, G: SeedableRng, E> StochasiticCompute<I, O, F, G, E>
where
    F: Fn(&mut G, I) -> Result<O>,
{
    /// Create a new stochastic cached computation
    pub fn new(function: F) -> Self {
        Self {
            function,
            rng: G::from_rng(&mut rand::rng()),
            _phantom: PhantomData,
        }
    }
}

impl<I, O, F, G: SeedableRng, E> StochasiticCompute<I, O, F, G, E> {
    /// Reseed the internal RNG with a new seed
    pub fn reseed(&mut self, seed: G::Seed) {
        self.rng = G::from_seed(seed);
    }
}

impl<I, O, F: Clone, G: SeedableRng, E> Clone for StochasiticCompute<I, O, F, G, E> {
    fn clone(&self) -> Self {
        Self {
            function: self.function.clone(),
            rng: G::from_rng(&mut rand::rng()),
            _phantom: PhantomData,
        }
    }
}

impl<F, I, O, G, C, E> Compute<C> for StochasiticCompute<I, O, F, G, E>
where
    F: for<'g> Fn(&'g mut G, I) -> Result<O>,
    G: Rng,
    C: CacheStore<E>,
    E: CodecEngine + Encode<I> + Encode<usize> + Encode<O> + Decode<O>,
{
    type Engine = E;
    type Input = (usize, I);
    type Output = O;

    fn execute(
        &mut self,
        input: Self::Input,
        cache: &C,
        engine: &mut Self::Engine,
    ) -> Result<Self::Output> {
        let (i, input) = input;
        let mut sig = engine.encode(&input).to_vec();
        let index = engine.encode(&i).to_vec();
        sig.extend_from_slice(&index);

        fetch_or_execute(&sig, cache, engine, || {
            (self.function)(&mut self.rng, input)
        })
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
        thread::{sleep, spawn},
        time::Duration,
    };

    use rand::rngs::SmallRng;
    use rayon::prelude::*;

    use super::*;
    use crate::HashMapStore;

    type Engine = crate::BitcodeCodec;

    #[test]
    fn test_pure() -> Result<()> {
        let compute: PureCompute<usize, usize, _, Engine> = PureCompute::new(|i| Ok(i + 1usize));
        let cache = HashMapStore::<RandomState>::default();

        let results = compute
            .execute_many(
                &cache,
                Arc::new(AtomicBool::new(false)),
                (0..100).into_par_iter(),
            )?
            .collect::<Result<Vec<usize>>>()?;

        assert_eq!(results, (0..100).map(|i| i + 1).collect::<Vec<usize>>());

        Ok(())
    }

    #[test]
    fn test_interrupt() -> Result<()> {
        let compute: PureCompute<usize, usize, _, Engine> = PureCompute::new(|i| {
            sleep(Duration::from_millis(100));
            Ok(i + 1usize)
        });
        let cache = HashMapStore::<RandomState>::default();

        let interrupted = Arc::new(AtomicBool::new(false));
        let interrupted_clone = interrupted.clone();

        let handle = spawn(move || {
            compute
                .execute_many(&cache, interrupted, (0..100).into_par_iter())?
                .collect::<Result<Vec<usize>>>()
        });

        sleep(Duration::from_millis(50));
        interrupted_clone.store(true, Ordering::Relaxed);

        let results = handle.join().expect("Failed to join thread");

        assert!(matches!(results.unwrap_err(), crate::Error::Interrupted));

        Ok(())
    }

    #[test]
    fn test_pure_caching() -> Result<()> {
        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count_clone = call_count.clone();

        let compute: PureCompute<usize, usize, _, Engine> = PureCompute::new(move |i| {
            call_count_clone.fetch_add(1, Ordering::SeqCst);
            Ok(i * 2)
        });

        let cache = HashMapStore::<RandomState>::default();
        let n_inputs = 5;

        // First execution - should call function for each input
        let results1 = compute
            .execute_many(
                &cache,
                Arc::new(AtomicBool::new(false)),
                (0..n_inputs).into_par_iter(),
            )?
            .collect::<Result<Vec<usize>>>()?;

        let expected: Vec<usize> = (0..n_inputs).map(|i| i * 2).collect();
        assert_eq!(results1, expected);
        assert_eq!(call_count.load(Ordering::SeqCst), n_inputs);

        // Second execution - should use cached results (no additional calls)
        let results2 = compute
            .execute_many(
                &cache,
                Arc::new(AtomicBool::new(false)),
                (0..n_inputs).into_par_iter(),
            )?
            .collect::<Result<Vec<usize>>>()?;

        assert_eq!(results2, expected);
        // Should still be the same number of calls (cached)
        assert_eq!(call_count.load(Ordering::SeqCst), n_inputs);

        Ok(())
    }

    #[test]
    fn test_stochastic() -> Result<()> {
        let compute: StochasiticCompute<usize, usize, _, SmallRng, Engine> =
            StochasiticCompute::new(|rng: &mut SmallRng, i| {
                // Sleep to test if execute many return results in the same order as the input
                sleep(Duration::from_millis(rng.random_range(0..100)));
                Ok(i + 1)
            });
        let n_input = rayon::current_num_threads();
        let cache = HashMapStore::<RandomState>::default();

        let inputs = (0..n_input).into_par_iter().map(|i| (i, i));
        let results = compute
            .execute_many(
                &cache,
                Arc::new(AtomicBool::new(false)),
                inputs.into_par_iter(),
            )?
            .collect::<Result<Vec<usize>>>()?;

        assert_eq!(results, (0..n_input).map(|i| i + 1).collect::<Vec<usize>>());

        Ok(())
    }

    #[test]
    fn test_stochastic_cache() -> Result<()> {
        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count_clone = call_count.clone();

        let compute: StochasiticCompute<usize, usize, _, SmallRng, Engine> =
            StochasiticCompute::new(move |_: &mut SmallRng, i| {
                call_count_clone.fetch_add(1, Ordering::SeqCst);
                Ok(i * 3)
            });

        let cache = HashMapStore::<RandomState>::default();
        let n_inputs = 20;

        // First execution - should call function for each input
        let inputs = (0..n_inputs).into_par_iter().map(|i| (i, i));
        let results1 = compute
            .execute_many(&cache, Arc::new(AtomicBool::new(false)), inputs)?
            .collect::<Result<Vec<usize>>>()?;

        let expected: Vec<usize> = (0..n_inputs).map(|i| i * 3).collect();
        assert_eq!(results1, expected);
        assert_eq!(call_count.load(Ordering::SeqCst), n_inputs);

        // Second execution - should use cached results
        let inputs = (0..n_inputs).into_par_iter().map(|i| (i, i));
        let results2 = compute
            .execute_many(&cache, Arc::new(AtomicBool::new(false)), inputs)?
            .collect::<Result<Vec<usize>>>()?;

        assert_eq!(results2, expected);
        // Should still be the same number of calls (cached)
        assert_eq!(call_count.load(Ordering::SeqCst), n_inputs);

        Ok(())
    }

    #[test]
    fn test_stochastic_seeding() -> Result<()> {
        let mut compute1 =
            StochasiticCompute::<usize, usize, _, SmallRng, Engine>::new(|rng, i| {
                Ok(i + rng.random_range(0..100))
            });

        let mut compute2 =
            StochasiticCompute::<usize, usize, _, SmallRng, Engine>::new(|rng, i| {
                Ok(i + rng.random_range(0..100))
            });

        // Set same seed for both computations
        let seed = [42; 32];
        compute1.reseed(seed);
        compute2.reseed(seed);

        let cache1 = HashMapStore::<RandomState>::default();
        let cache2 = HashMapStore::<RandomState>::default();

        // Same inputs with same seed should produce same results
        let inputs = [(0, 5), (1, 10), (2, 15)];

        let mut engine1 = Engine::default();
        let mut engine2 = Engine::default();

        let results1: Result<Vec<_>> = inputs
            .iter()
            .map(|&input| compute1.execute(input, &cache1, &mut engine1))
            .collect();

        let results2: Result<Vec<_>> = inputs
            .iter()
            .map(|&input| compute2.execute(input, &cache2, &mut engine2))
            .collect();

        assert_eq!(results1?, results2?);

        Ok(())
    }
}
