use std::marker::PhantomData;

use super::DependentStochasticInput;
use crate::{
    Compute,
    cache::{Cache, CacheProvider, CanonicalEncode, CloneShared},
    compute::{
        NoFunction,
        stream::{MultiStreams, SeedSource, SingleStream, StreamSpec},
    },
    error::Result,
    identity::{ComputationId, ComputationPath},
};

type BuildStochasticTransform<CP, U, T, O, S> =
    StochasticTransform<U, <CP as CacheProvider<O>>::Cache, T, O, S>;

fn upstream_encoded<P, I>(encoded: &[u8]) -> &[u8]
where
    P: CanonicalEncode,
    I: CanonicalEncode,
{
    &encoded[P::SIZE..P::SIZE + I::SIZE]
}

/// Internal function wrapper for non-parameterized stochastic transforms.
#[doc(hidden)]
#[derive(Clone)]
pub struct StochasticTransformFunction<F> {
    function: F,
}

/// Internal function wrapper for parameterized stochastic transforms.
#[doc(hidden)]
pub struct ParamStochasticTransformFunction<F, P> {
    function: F,
    _param: PhantomData<P>,
}

impl<F: Clone, P> Clone for ParamStochasticTransformFunction<F, P> {
    fn clone(&self) -> Self {
        Self {
            function: self.function.clone(),
            _param: PhantomData,
        }
    }
}

/// Internal bridge from encoded input to transform call behavior.
#[doc(hidden)]
pub trait StochasticTransformFn<U, Rng, O>
where
    U: Compute,
{
    type Input: CanonicalEncode;
    type Param;

    fn split_input(input: Self::Input) -> (U::Input, Self::Param);
    fn upstream_encoded(encoded: &[u8]) -> &[u8];
    fn call(&self, rng: &mut Rng, upstream: U::Output, param: Self::Param) -> Result<O>;
}

impl<U, F, R, O> StochasticTransformFn<U, R, O> for StochasticTransformFunction<F>
where
    U: Compute,
    F: Fn(&mut R, U::Output) -> Result<O>,
{
    type Input = DependentStochasticInput<(), U::Input>;
    type Param = ();

    fn split_input(input: Self::Input) -> (U::Input, Self::Param) {
        (input.source, input.param)
    }

    fn upstream_encoded(encoded: &[u8]) -> &[u8] {
        upstream_encoded::<(), U::Input>(encoded)
    }

    fn call(&self, rng: &mut R, upstream: U::Output, _: Self::Param) -> Result<O> {
        (self.function)(rng, upstream)
    }
}

impl<U, F, R, P, O> StochasticTransformFn<U, R, O> for ParamStochasticTransformFunction<F, P>
where
    U: Compute,
    F: Fn(&mut R, U::Output, P) -> Result<O>,
    P: CanonicalEncode,
{
    type Input = DependentStochasticInput<P, U::Input>;
    type Param = P;

    fn split_input(input: Self::Input) -> (U::Input, Self::Param) {
        (input.source, input.param)
    }

    fn upstream_encoded(encoded: &[u8]) -> &[u8] {
        upstream_encoded::<P, U::Input>(encoded)
    }

    fn call(&self, rng: &mut R, upstream: U::Output, param: Self::Param) -> Result<O> {
        (self.function)(rng, upstream, param)
    }
}

/// Stochastic transform derived from an upstream compute node.
pub struct StochasticTransform<U, C, F, O, S> {
    path: ComputationPath,
    upstream: U,
    cache: C,
    seed: S,
    transform: F,
    _output: PhantomData<O>,
}

impl<U, C, F, O, S> Clone for StochasticTransform<U, C, F, O, S>
where
    U: Clone,
    C: CloneShared,
    S: Clone,
    F: Clone,
{
    fn clone(&self) -> Self {
        Self {
            path: self.path.clone(),
            upstream: self.upstream.clone(),
            cache: self.cache.clone_shared(),
            seed: self.seed.clone(),
            transform: self.transform.clone(),
            _output: PhantomData,
        }
    }
}

impl<U, C, F, O, S> Compute for StochasticTransform<U, C, F, O, S>
where
    U: Compute,
    C: Cache<O>,
    S: SeedSource,
    F: StochasticTransformFn<U, S::Rng, O>,
    F::Input: CanonicalEncode,
{
    type Input = F::Input;
    type Output = O;

    fn computation_path(&self) -> &ComputationPath {
        &self.path
    }

    fn execute_with_encoded_input(
        &mut self,
        input: Self::Input,
        encoded: &[u8],
    ) -> Result<Self::Output> {
        let (upstream_input, param) = F::split_input(input);
        let upstream_encoded = F::upstream_encoded(encoded);
        let cache = &mut self.cache;
        let upstream = &mut self.upstream;
        let seed = &self.seed;
        let transform = &self.transform;
        cache.fetch_or_execute(encoded, || {
            let upstream = upstream.execute_with_encoded_input(upstream_input, upstream_encoded)?;
            let mut rng = seed.make_rng(encoded);
            transform.call(&mut rng, upstream, param)
        })
    }
}

/// Builder for a stochastic transform.
pub struct StochasticTransformBuilder<U, F = NoFunction, O = (), S = SingleStream, CP = ()>
where
    U: Compute,
{
    pub(super) upstream: U,
    pub(super) id: ComputationId,
    pub(super) streams: S,
    pub(super) transform: F,
    pub(super) provider: CP,
    pub(super) _output: PhantomData<O>,
}

impl<U, F, O, S, CP> StochasticTransformBuilder<U, F, O, S, CP>
where
    U: Compute,
{
    /// Replace the cache provider for this transform result.
    ///
    /// If this is not called, the builder uses `()` and executes without caching this transform.
    pub fn cache<NextCP>(self, provider: NextCP) -> StochasticTransformBuilder<U, F, O, S, NextCP> {
        StochasticTransformBuilder {
            upstream: self.upstream,
            id: self.id,
            streams: self.streams,
            transform: self.transform,
            provider,
            _output: PhantomData,
        }
    }
}

impl<U, CP> StochasticTransformBuilder<U, NoFunction, (), SingleStream, CP>
where
    U: Compute,
{
    /// Replace the default single RNG stream with named streams.
    pub fn streams<const N: usize, I: Into<MultiStreams<N>>>(
        self,
        variables: I,
    ) -> StochasticTransformBuilder<U, NoFunction, (), MultiStreams<N>, CP> {
        StochasticTransformBuilder {
            upstream: self.upstream,
            id: self.id,
            streams: variables.into(),
            transform: self.transform,
            provider: self.provider,
            _output: PhantomData,
        }
    }
}

impl<U, O, S, CP> StochasticTransformBuilder<U, NoFunction, O, S, CP>
where
    U: Compute,
    S: StreamSpec,
{
    /// Replace the stochastic transform function.
    pub fn function<NextF, NextO>(
        self,
        transform: NextF,
    ) -> StochasticTransformBuilder<U, StochasticTransformFunction<NextF>, NextO, S, CP>
    where
        NextF: Fn(&mut <S::Seed as SeedSource>::Rng, U::Output) -> Result<NextO>,
    {
        StochasticTransformBuilder {
            upstream: self.upstream,
            id: self.id,
            streams: self.streams,
            transform: StochasticTransformFunction {
                function: transform,
            },
            provider: self.provider,
            _output: PhantomData,
        }
    }

    /// Replace the parameterized stochastic transform function.
    pub fn function_with_param<NextP, NextF, NextO>(
        self,
        transform: NextF,
    ) -> StochasticTransformBuilder<U, ParamStochasticTransformFunction<NextF, NextP>, NextO, S, CP>
    where
        NextF: Fn(&mut <S::Seed as SeedSource>::Rng, U::Output, NextP) -> Result<NextO>,
    {
        StochasticTransformBuilder {
            upstream: self.upstream,
            id: self.id,
            streams: self.streams,
            transform: ParamStochasticTransformFunction {
                function: transform,
                _param: PhantomData,
            },
            provider: self.provider,
            _output: PhantomData,
        }
    }
}

impl<U, T, O, S, CP> StochasticTransformBuilder<U, T, O, S, CP>
where
    U: Compute,
    S: StreamSpec,
    T: StochasticTransformFn<U, <S::Seed as SeedSource>::Rng, O>,
    CP: CacheProvider<O>,
{
    /// Bind providers recursively and build the stochastic transform.
    pub fn build(self) -> Result<BuildStochasticTransform<CP, U, T, O, S::Seed>> {
        let path = self.upstream.computation_path().child(self.id);
        let cache = self.provider.bind(&path)?;
        let seed = self.streams.derive_seed(&path);
        Ok(StochasticTransform {
            path,
            upstream: self.upstream,
            cache,
            seed,
            transform: self.transform,
            _output: PhantomData,
        })
    }
}
