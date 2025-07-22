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
