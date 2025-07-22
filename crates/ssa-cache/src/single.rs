use std::marker::PhantomData;

use rand::{Rng, SeedableRng};

use crate::{CacheStore, Cacheable, Compute, Encodeable, Result};

#[derive(Debug)]
pub struct PureCompute<I, O, F> {
    function: F,
    _phantom: PhantomData<(I, O)>,
}

impl<I, O, F: Fn(I) -> O> PureCompute<I, O, F> {
    /// Create a new pure (deterministic) cached computation
    pub fn new(function: F) -> Self {
        Self {
            function,
            _phantom: PhantomData,
        }
    }
}

impl<I, O, F: Clone> Clone for PureCompute<I, O, F> {
    fn clone(&self) -> Self {
        Self {
            function: self.function.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<I, O, F, C> Compute<[u8], C> for PureCompute<I, O, F>
where
    I: Encodeable,
    O: Cacheable<Buffer = I::Buffer>,
    F: Fn(I) -> O,
    C: CacheStore<[u8]>,
{
    type Input = I;
    type Output = O;

    fn raw_execute(&mut self, input: I) -> Self::Output {
        (self.function)(input)
    }

    fn execute(
        &mut self,
        input: Self::Input,
        cache: &C,
        buffer: &mut <Self::Output as Encodeable>::Buffer,
    ) -> Result<Self::Output> {
        let sig = input.encode(buffer).to_owned();
        self.execute_with_sig(&sig, input, cache, buffer)
    }
}

#[derive(Debug)]
pub struct StochasiticCompute<I, O, F, G> {
    function: F,
    rng: G,
    _phantom: PhantomData<(I, O)>,
}

impl<I, O, F: Clone, G: SeedableRng> StochasiticCompute<I, O, F, G>
where
    F: Fn(&mut G, I) -> O,
    G: Rng + SeedableRng,
{
    /// Create a new stochastic cached computation
    pub fn new(function: F) -> Self {
        Self {
            function,
            rng: G::from_rng(&mut rand::rng()),
            _phantom: PhantomData,
        }
    }

    /// Reseed the internal RNG with a new seed
    pub fn reseed(&mut self, seed: G::Seed) {
        self.rng = G::from_seed(seed);
    }
}

impl<I, O, F: Clone, G: SeedableRng> Clone for StochasiticCompute<I, O, F, G> {
    fn clone(&self) -> Self {
        Self {
            function: self.function.clone(),
            rng: G::from_rng(&mut rand::rng()),
            _phantom: PhantomData,
        }
    }
}

impl<F, I, O, G, C> Compute<[u8], C> for StochasiticCompute<I, O, F, G>
where
    F: for<'g> Fn(&'g mut G, I) -> O,
    G: Rng,
    I: Encodeable,
    O: Cacheable<Buffer = I::Buffer>,
    C: CacheStore<[u8]>,
{
    type Input = (usize, I);
    type Output = O;

    fn raw_execute(&mut self, (_, input): Self::Input) -> Self::Output {
        (self.function)(&mut self.rng, input)
    }

    fn execute(
        &mut self,
        input: Self::Input,
        cache: &C,
        buffer: &mut <Self::Output as Encodeable>::Buffer,
    ) -> Result<Self::Output> {
        let (i, input) = input;
        let sig = [input.encode(buffer), &i.to_le_bytes()].concat();
        self.execute_with_sig(&sig, (i, input), cache, buffer)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        hash::RandomState,
        thread::sleep,
        time::{Duration, Instant},
    };

    use rand::rngs::SmallRng;
    use rayon::prelude::*;

    use super::*;
    use crate::HashMapStore;

    #[test]
    fn test_pure() -> Result<()> {
        let compute = PureCompute::new(|i| i + 1usize);
        let cache = HashMapStore::<RandomState>::default();

        // Test fresh results
        let results = compute
            .execute_many(&cache, (0..100).into_par_iter())?
            .collect::<Result<Vec<usize>>>()?;

        assert_eq!(results, (0..100).map(|i| i + 1).collect::<Vec<usize>>());

        // Test results are cached
        let results = compute
            .execute_many(&cache, (0..100).into_par_iter())?
            .collect::<Result<Vec<usize>>>()?;

        assert_eq!(results, (0..100).map(|i| i + 1).collect::<Vec<usize>>());

        Ok(())
    }

    #[test]
    fn test_stochastic() -> Result<()> {
        let compute = StochasiticCompute::new(|rng: &mut SmallRng, i| {
            // Sleep to test if execute many return results in the same order as the input
            sleep(Duration::from_millis(rng.random_range(0..100)));
            i + 1
        });
        let n_input = rayon::current_num_threads() * 10;
        let cache = HashMapStore::<RandomState>::default();

        // Test fresh results
        let inputs = (0..n_input).into_par_iter().map(|i| (i, i));
        let results = compute
            .execute_many(&cache, inputs.into_par_iter())?
            .collect::<Result<Vec<usize>>>()?;

        assert_eq!(results, (0..n_input).map(|i| i + 1).collect::<Vec<usize>>());

        // Test cache hits
        let inputs = (0..n_input).into_par_iter().map(|i| (i, i));
        let start = Instant::now();
        let results = compute
            .execute_many(&cache, inputs.into_par_iter())?
            .collect::<Result<Vec<usize>>>()?;
        let time = start.elapsed();

        // If cache missing, we need around 500 ms + overhead
        assert_eq!(results, (0..n_input).map(|i| i + 1).collect::<Vec<usize>>());
        assert!(time.as_millis() < 200);

        Ok(())
    }
}
