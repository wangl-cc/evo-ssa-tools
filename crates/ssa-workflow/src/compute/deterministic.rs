use std::marker::PhantomData;

use crate::{
    Compute, Result,
    cache::{Cache, CacheProvider, CanonicalEncode, CloneShared},
    compute::NoFunction,
    identity::{ComputationId, ComputationPath},
};

/// Deterministic compute node.
///
/// `DeterministicTask` is a memoized `input -> output` mapping where the output is a pure
/// function of the input (no randomness, no external state).
///
/// # When to use
///
/// - The output depends only on the input value.
/// - You want to reuse results across repeated calls / parameter sweeps.
/// - You want parallel batch execution via [`Compute::with_inputs`].
///
/// # Example
///
/// ```rust
/// # use ssa_workflow::prelude::*;
/// # fn main() -> ssa_workflow::error::Result<()> {
/// let task = DeterministicTask::builder("abs-v1")
///     .function(|i: i32| Ok(i.abs()))
///     .cache(ManagedHashCache::<i32>::default())
///     .build()?;
/// let results = task
///     .with_inputs(0..10)
///     .collect::<ssa_workflow::Result<Vec<_>>>()?;
/// # Ok(())
/// # }
/// ```
pub struct DeterministicTask<C, I, O, F> {
    path: ComputationPath,
    cache: C,
    function: F,
    _phantom: PhantomData<(I, O)>,
}

impl DeterministicTask<(), (), (), NoFunction> {
    /// Start a deterministic task builder for a computation id.
    pub fn builder(id: impl Into<ComputationId>) -> DeterministicTaskBuilder {
        DeterministicTaskBuilder {
            id: id.into(),
            function: NoFunction,
            provider: (),
            _phantom: PhantomData,
        }
    }
}

impl<C: CloneShared, I, O, F: Clone> Clone for DeterministicTask<C, I, O, F> {
    fn clone(&self) -> Self {
        Self {
            path: self.path.clone(),
            cache: self.cache.clone_shared(),
            function: self.function.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<C, I, O, F> Compute for DeterministicTask<C, I, O, F>
where
    F: Fn(I) -> Result<O>,
    C: Cache<O>,
    I: CanonicalEncode,
{
    type Input = I;
    type Output = O;

    fn computation_path(&self) -> &ComputationPath {
        &self.path
    }

    fn execute_with_encoded_input(
        &mut self,
        input: Self::Input,
        encoded: &[u8],
    ) -> Result<Self::Output> {
        let cache = &mut self.cache;
        let function = &self.function;
        cache.fetch_or_execute(encoded, || function(input))
    }
}

/// Builder for deterministic tasks.
pub struct DeterministicTaskBuilder<I = (), O = (), F = NoFunction, P = ()> {
    id: ComputationId,
    function: F,
    provider: P,
    _phantom: PhantomData<(I, O)>,
}

impl<I, O, F, P> DeterministicTaskBuilder<I, O, F, P> {
    /// Replace the compute function for this task.
    pub fn function<NextI, NextO, NextF>(
        self,
        function: NextF,
    ) -> DeterministicTaskBuilder<NextI, NextO, NextF, P>
    where
        NextF: Fn(NextI) -> Result<NextO>,
    {
        DeterministicTaskBuilder {
            id: self.id,
            function,
            provider: self.provider,
            _phantom: PhantomData,
        }
    }

    /// Replace the cache provider for this task result.
    ///
    /// If this is not called, the builder uses `()` and executes without caching this task.
    pub fn cache<NextP>(self, provider: NextP) -> DeterministicTaskBuilder<I, O, F, NextP> {
        DeterministicTaskBuilder {
            id: self.id,
            function: self.function,
            provider,
            _phantom: PhantomData,
        }
    }
}

impl<I, O, F, P> DeterministicTaskBuilder<I, O, F, P>
where
    F: Fn(I) -> Result<O>,
{
    /// Bind the provider and build the deterministic task.
    pub fn build(self) -> Result<DeterministicTask<P::Cache, I, O, F>>
    where
        P: CacheProvider<O>,
        I: CanonicalEncode,
    {
        let path = ComputationPath::root(self.id);
        let cache = self.provider.bind(&path)?;
        Ok(DeterministicTask {
            path,
            cache,
            function: self.function,
            _phantom: PhantomData,
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
    use crate::cache::memory::ManagedHashCache;

    #[test]
    fn test_deterministic_caching() -> Result<()> {
        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count_clone = call_count.clone();

        let compute = DeterministicTask::builder("test-deterministic-caching-v1")
            .function(move |i: usize| {
                call_count_clone.fetch_add(1, Ordering::SeqCst);
                Ok(i * 3)
            })
            .cache(ManagedHashCache::<usize>::default())
            .build()?;

        let n_inputs = 5;

        let results1 = compute
            .with_inputs(0..n_inputs)
            .collect::<Result<Vec<_>>>()?;

        let expected: Vec<usize> = (0..n_inputs).map(|i| i * 3).collect();
        assert_eq!(results1, expected);
        assert_eq!(call_count.load(Ordering::SeqCst), n_inputs);

        let results2 = compute
            .with_inputs(0..n_inputs)
            .collect::<Result<Vec<_>>>()?;

        assert_eq!(results2, expected);
        assert_eq!(call_count.load(Ordering::SeqCst), n_inputs);

        Ok(())
    }

    #[test]
    fn test_deterministic_error_propagation() -> Result<()> {
        let mut compute = DeterministicTask::builder("test-deterministic-error-v1")
            .function(|i: usize| {
                if i == 5 {
                    Err(crate::error::Error::Interrupted)
                } else {
                    Ok(i * 2)
                }
            })
            .cache(ManagedHashCache::<usize>::default())
            .build()?;

        let result = compute.execute_one(3)?;
        assert_eq!(result, 6);

        let result = compute.execute_one(5);
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_deterministic_execution_order() -> Result<()> {
        let compute = DeterministicTask::builder("test-deterministic-order-v1")
            .function(|i: usize| Ok(i + 100))
            .cache(ManagedHashCache::<usize>::default())
            .build()?;

        let inputs = vec![3usize, 0, 2, 1, 4];
        let results = compute
            .with_inputs(inputs.clone())
            .collect::<Result<Vec<_>>>()?;

        assert_eq!(
            results,
            inputs.into_iter().map(|i| i + 100).collect::<Vec<usize>>()
        );

        Ok(())
    }

    #[test]
    fn batch_collect_naturally_returns_the_real_item_error() {
        let compute = DeterministicTask::builder("test-deterministic-real-error-v1")
            .function(|i: usize| {
                if i == 0 {
                    Err(crate::Error::Compute(
                        std::io::Error::other("real failure").into(),
                    ))
                } else {
                    Ok(i)
                }
            })
            .build()
            .expect("no-cache deterministic task should build");

        let error = compute
            .with_inputs(0..128usize)
            .collect::<Result<Vec<_>>>()
            .unwrap_err();

        assert!(matches!(error, crate::Error::Compute(_)));
    }

    #[test]
    fn builder_defaults_to_no_cache() -> Result<()> {
        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count_clone = call_count.clone();

        let mut compute = DeterministicTask::builder("test-no-cache-deterministic-v1")
            .function(move |i: usize| {
                call_count_clone.fetch_add(1, Ordering::SeqCst);
                Ok(i * 11)
            })
            .build()?;

        assert_eq!(compute.execute_one(3)?, 33);
        assert_eq!(compute.execute_one(3)?, 33);
        assert_eq!(call_count.load(Ordering::SeqCst), 2);
        Ok(())
    }

    #[derive(Clone, Copy)]
    struct FailingProvider;

    impl<T> CacheProvider<T> for FailingProvider {
        type Cache = ();

        fn bind(self, _: &ComputationPath) -> Result<Self::Cache> {
            Err(crate::Error::Compute(
                std::io::Error::other("bind failed").into(),
            ))
        }
    }

    #[test]
    fn build_propagates_cache_provider_error() {
        let result = DeterministicTask::builder("test-bind-failure-v1")
            .function(|input: u8| Ok(input))
            .cache(FailingProvider)
            .build();

        assert!(matches!(result, Err(crate::Error::Compute(_))));
    }
}
