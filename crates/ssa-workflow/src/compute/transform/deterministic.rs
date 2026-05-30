use std::marker::PhantomData;

use super::DependentInput;
use crate::{
    Compute,
    cache::{Cache, CacheProvider, CanonicalEncode, CloneShared},
    compute::NoFunction,
    error::Result,
    identity::{ComputationId, ComputationPath},
};

/// Internal function wrapper for non-parameterized deterministic transforms.
#[doc(hidden)]
#[derive(Clone)]
pub struct TransformFunction<F> {
    function: F,
}

/// Internal function wrapper for parameterized deterministic transforms.
#[doc(hidden)]
pub struct ParamTransformFunction<F, P> {
    function: F,
    _param: PhantomData<P>,
}

impl<F: Clone, P> Clone for ParamTransformFunction<F, P> {
    fn clone(&self) -> Self {
        Self {
            function: self.function.clone(),
            _param: PhantomData,
        }
    }
}

/// Internal bridge from encoded input to transform call behavior.
#[doc(hidden)]
pub trait DeterministicTransformFn<U, O>
where
    U: Compute,
{
    type Input: CanonicalEncode;
    type Param;

    fn split_input(input: Self::Input) -> (U::Input, Self::Param);
    fn upstream_encoded(encoded: &[u8]) -> &[u8];
    fn call(&self, upstream: U::Output, param: Self::Param) -> Result<O>;
}

impl<U, F, O> DeterministicTransformFn<U, O> for TransformFunction<F>
where
    U: Compute,
    F: Fn(U::Output) -> Result<O>,
{
    type Input = U::Input;
    type Param = ();

    fn split_input(input: Self::Input) -> (U::Input, Self::Param) {
        (input, ())
    }

    fn upstream_encoded(encoded: &[u8]) -> &[u8] {
        encoded
    }

    fn call(&self, upstream: U::Output, _: Self::Param) -> Result<O> {
        (self.function)(upstream)
    }
}

impl<U, F, P, O> DeterministicTransformFn<U, O> for ParamTransformFunction<F, P>
where
    U: Compute,
    F: Fn(U::Output, P) -> Result<O>,
    P: CanonicalEncode,
{
    type Input = DependentInput<P, U::Input>;
    type Param = P;

    fn split_input(input: Self::Input) -> (U::Input, Self::Param) {
        (input.source, input.param)
    }

    fn upstream_encoded(encoded: &[u8]) -> &[u8] {
        let start = P::SIZE;
        &encoded[start..start + U::Input::SIZE]
    }

    fn call(&self, upstream: U::Output, param: Self::Param) -> Result<O> {
        (self.function)(upstream, param)
    }
}

/// Dependent deterministic transform node.
///
/// A [`Transform`] runs an upstream computation and then applies a local deterministic function to
/// the upstream output. Its computation path extends the upstream path with the transform id, so
/// changing an upstream task selects a different cache namespace for the transform result.
pub struct Transform<U, C, F, O> {
    path: ComputationPath,
    upstream: U,
    cache: C,
    transform: F,
    _output: PhantomData<O>,
}

impl<U: Clone, C: CloneShared, F: Clone, O> Clone for Transform<U, C, F, O> {
    fn clone(&self) -> Self {
        Self {
            path: self.path.clone(),
            upstream: self.upstream.clone(),
            cache: self.cache.clone_shared(),
            transform: self.transform.clone(),
            _output: PhantomData,
        }
    }
}

impl<U, C, F, O> Compute for Transform<U, C, F, O>
where
    U: Compute,
    C: Cache<O>,
    F: DeterministicTransformFn<U, O>,
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
        let transform = &self.transform;
        cache.fetch_or_execute(encoded, || {
            let intermediate =
                upstream.execute_with_encoded_input(upstream_input, upstream_encoded)?;
            transform.call(intermediate, param)
        })
    }
}

/// Builder for a deterministic transform.
pub struct TransformBuilder<U, F = NoFunction, O = (), CP = ()>
where
    U: Compute,
{
    pub(super) upstream: U,
    pub(super) id: ComputationId,
    pub(super) transform: F,
    pub(super) provider: CP,
    pub(super) _output: PhantomData<O>,
}

impl<U, F, O, CP> TransformBuilder<U, F, O, CP>
where
    U: Compute,
{
    /// Replace the cache provider for this transform result.
    ///
    /// If this is not called, the builder uses `()` and executes without caching this transform.
    pub fn cache<NextCP>(self, provider: NextCP) -> TransformBuilder<U, F, O, NextCP> {
        TransformBuilder {
            upstream: self.upstream,
            id: self.id,
            transform: self.transform,
            provider,
            _output: PhantomData,
        }
    }
}

impl<U, O, CP> TransformBuilder<U, NoFunction, O, CP>
where
    U: Compute,
{
    /// Replace the transform function.
    pub fn function<NextF, NextO>(
        self,
        transform: NextF,
    ) -> TransformBuilder<U, TransformFunction<NextF>, NextO, CP>
    where
        NextF: Fn(U::Output) -> Result<NextO>,
    {
        TransformBuilder {
            upstream: self.upstream,
            id: self.id,
            transform: TransformFunction {
                function: transform,
            },
            provider: self.provider,
            _output: PhantomData,
        }
    }

    /// Replace the parameterized transform function.
    pub fn function_with_param<P, NextF, NextO>(
        self,
        transform: NextF,
    ) -> TransformBuilder<U, ParamTransformFunction<NextF, P>, NextO, CP>
    where
        NextF: Fn(U::Output, P) -> Result<NextO>,
    {
        TransformBuilder {
            upstream: self.upstream,
            id: self.id,
            transform: ParamTransformFunction {
                function: transform,
                _param: PhantomData,
            },
            provider: self.provider,
            _output: PhantomData,
        }
    }
}

impl<U, F, O, CP> TransformBuilder<U, F, O, CP>
where
    U: Compute,
    F: DeterministicTransformFn<U, O>,
    F::Input: CanonicalEncode,
    CP: CacheProvider<O>,
{
    /// Bind the cache provider and build the transform.
    pub fn build(self) -> Result<Transform<U, CP::Cache, F, O>> {
        let path = self.upstream.computation_path().child(self.id);
        let cache = self.provider.bind::<F::Input>(&path)?;
        Ok(Transform {
            path,
            upstream: self.upstream,
            cache,
            transform: self.transform,
            _output: PhantomData,
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

    use super::*;
    use crate::{
        Compute, Result,
        cache::memory::ManagedHashCache,
        compute::{DeterministicTask, TransformExt},
    };

    #[test]
    fn transform_builder_composes_built_nodes() -> Result<()> {
        let source_calls = Arc::new(AtomicUsize::new(0));
        let source = DeterministicTask::builder("test-source-v1")
            .function({
                let source_calls = Arc::clone(&source_calls);
                move |input: u16| {
                    source_calls.fetch_add(1, Ordering::SeqCst);
                    Ok(input * 2)
                }
            })
            .cache(ManagedHashCache::<u16>::default())
            .build()?;
        let mut transform = source
            .transform("test-plus-one-v1")
            .function(|intermediate| Ok(intermediate + 1))
            .cache(ManagedHashCache::<u16>::default())
            .build()?;

        assert_eq!(transform.execute_one(7)?, 15);
        assert_eq!(transform.execute_one(7)?, 15);
        assert_eq!(source_calls.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[test]
    fn transform_builder_supports_parameterized_transforms() -> Result<()> {
        let source_calls = Arc::new(AtomicUsize::new(0));
        let source = DeterministicTask::builder("test-param-source-v1")
            .function({
                let source_calls = Arc::clone(&source_calls);
                move |input: u16| {
                    source_calls.fetch_add(1, Ordering::SeqCst);
                    Ok(input * 2)
                }
            })
            .cache(ManagedHashCache::<u16>::default())
            .build()?;
        let mut transform = source
            .transform("test-add-param-v1")
            .function_with_param(|intermediate, param: u16| Ok(intermediate + param))
            .cache(ManagedHashCache::<u16>::default())
            .build()?;

        assert_eq!(transform.execute_one(DependentInput::new(3, 7))?, 17);
        assert_eq!(transform.execute_one(DependentInput::new(3, 7))?, 17);
        assert_eq!(source_calls.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[derive(Clone, Copy)]
    struct FailingProvider;

    impl<T> CacheProvider<T> for FailingProvider {
        type Cache = ();

        fn bind<I: CanonicalEncode>(self, _: &ComputationPath) -> Result<Self::Cache> {
            Err(crate::Error::Compute("bind failed".into()))
        }
    }

    #[test]
    fn transform_build_propagates_cache_provider_error() -> Result<()> {
        let source = DeterministicTask::builder("test-bind-source-v1")
            .function(|input: u8| Ok(input))
            .build()?;

        let result = source
            .transform("test-bind-transform-v1")
            .function(Ok)
            .cache(FailingProvider)
            .build();

        assert!(matches!(result, Err(crate::Error::Compute(_))));
        Ok(())
    }
}
