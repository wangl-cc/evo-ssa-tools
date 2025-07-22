use std::marker::PhantomData;

use rand::{Rng, SeedableRng};

use crate::{CacheStore, Cacheable, Compute, Encodeable, Result};

#[derive(Debug, Clone, Copy)]
pub struct ChainedSignature<S: ?Sized> {
    first: bool,
    signature: S,
}

impl<S: AsRef<[u8]>> ChainedSignature<S> {
    pub fn new(first: bool, signature: S) -> Self {
        Self { signature, first }
    }
}

impl<C1: CacheStore<[u8]>, C2: CacheStore<[u8]>> CacheStore<ChainedSignature<[u8]>> for (C1, C2) {
    fn fetch<T: Cacheable>(
        &self,
        sig: &ChainedSignature<[u8]>,
        buffer: &mut T::Buffer,
    ) -> Result<Option<T>> {
        if sig.first {
            self.0.fetch(&sig.signature, buffer)
        } else {
            self.1.fetch(&sig.signature, buffer)
        }
    }

    fn store<T: Cacheable>(
        &self,
        sig: &ChainedSignature<[u8]>,
        buffer: &mut T::Buffer,
        value: &T,
    ) -> Result<()> {
        if sig.first {
            self.0.store(&sig.signature, buffer, value)
        } else {
            self.1.store(&sig.signature, buffer, value)
        }
    }
}

/// A Chained Computation is a sequence of computations that each computation depends on the
/// output of the previous one.
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

impl<I, M, O, E, A, G, C1, C2> Compute<ChainedSignature<[u8]>, (C1, C2)>
    for ExpAnalysis<I, M, O, E, A, G>
where
    C1: CacheStore<[u8]>,
    C2: CacheStore<[u8]>,
    I: Encodeable,
    M: Cacheable<Buffer = I::Buffer>,
    O: Cacheable<Buffer = I::Buffer>,
    E: Fn(&mut G, I) -> M,
    A: Fn(M) -> O,
{
    type Input = (usize, I);
    type Output = O;

    fn raw_execute(&mut self, _: Self::Input) -> Self::Output {
        unimplemented!("Don't call this function directly. Use `execute` instead.")
    }

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
