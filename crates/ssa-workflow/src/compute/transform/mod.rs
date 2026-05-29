//! Dependent transform composition types.

use std::marker::PhantomData;

use super::NoFunction;
use crate::{Compute, identity::ComputationId};

pub mod deterministic;
mod input;
pub mod stochastic;

pub use deterministic::Transform;
pub use input::{DependentInput, DependentStochasticInput};

/// Extension trait for dependent transform construction.
pub trait TransformExt: Compute + Sized {
    /// Start a deterministic transform builder.
    fn transform(self, id: impl Into<ComputationId>) -> deterministic::TransformBuilder<Self> {
        deterministic::TransformBuilder {
            upstream: self,
            id: id.into(),
            transform: NoFunction,
            provider: (),
            _output: PhantomData,
        }
    }

    /// Start a single-stream stochastic transform builder without an explicit parameter.
    fn stochastic_transform(
        self,
        id: impl Into<ComputationId>,
    ) -> stochastic::StochasticTransformBuilder<Self> {
        stochastic::StochasticTransformBuilder {
            upstream: self,
            id: id.into(),
            streams: crate::compute::stream::SingleStream,
            transform: NoFunction,
            provider: (),
            _output: PhantomData,
        }
    }
}

impl<T: Compute> TransformExt for T {}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests;

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod inline_tests {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use super::TransformExt;
    use crate::{
        Result,
        cache::memory::ManagedHashCache,
        compute::{
            Compute, DependentStochasticInput, StochasticInput, deterministic::DeterministicTask,
            stochastic::StochasticTask,
        },
    };

    #[test]
    fn transform_ext_builds_dependent_transform_from_node() -> Result<()> {
        let mut transform = DeterministicTask::builder("test-source-v1")
            .function(|input: u16| Ok(input * 2))
            .cache(ManagedHashCache::<u16>::default())
            .build()?
            .transform("test-plus-one-v1")
            .function(|value| Ok(value + 1))
            .cache(ManagedHashCache::<u16>::default())
            .build()?;

        assert_eq!(transform.execute_one(5)?, 11);
        assert_eq!(transform.execute_one(5)?, 11);
        Ok(())
    }

    #[test]
    fn deterministic_transform_builder_defaults_to_no_cache() -> Result<()> {
        let source_calls = Arc::new(AtomicUsize::new(0));
        let transform_calls = Arc::new(AtomicUsize::new(0));

        let mut transform = DeterministicTask::builder("test-no-cache-source-v1")
            .function({
                let source_calls = Arc::clone(&source_calls);
                move |input: u16| {
                    source_calls.fetch_add(1, Ordering::SeqCst);
                    Ok(input * 2)
                }
            })
            .build()?
            .transform("test-no-cache-transform-v1")
            .function({
                let transform_calls = Arc::clone(&transform_calls);
                move |value| {
                    transform_calls.fetch_add(1, Ordering::SeqCst);
                    Ok(value + 1)
                }
            })
            .build()?;

        assert_eq!(transform.execute_one(5)?, 11);
        assert_eq!(transform.execute_one(5)?, 11);
        assert_eq!(source_calls.load(Ordering::SeqCst), 2);
        assert_eq!(transform_calls.load(Ordering::SeqCst), 2);
        Ok(())
    }

    #[test]
    fn stochastic_transform_builder_defaults_to_no_cache() -> Result<()> {
        let transform_calls = Arc::new(AtomicUsize::new(0));

        let mut transform = StochasticTask::builder("test-no-cache-stochastic-source-v1")
            .function(|rng, ()| Ok(rand::Rng::next_u64(rng)))
            .build()?
            .stochastic_transform("test-no-cache-stochastic-transform-v1")
            .function({
                let transform_calls = Arc::clone(&transform_calls);
                move |rng, value: u64| {
                    transform_calls.fetch_add(1, Ordering::SeqCst);
                    Ok(value ^ rand::Rng::next_u64(rng))
                }
            })
            .build()?;

        let input = DependentStochasticInput::new((), StochasticInput::new((), 5), 0);

        assert_eq!(
            transform.execute_one(input.clone())?,
            transform.execute_one(input)?
        );
        assert_eq!(transform_calls.load(Ordering::SeqCst), 2);
        Ok(())
    }
}
