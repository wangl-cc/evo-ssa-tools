//! Concrete compute node implementations.

pub mod deterministic;
pub mod execution;
pub mod stochastic;
pub mod stream;
pub mod transform;

pub use deterministic::DeterministicTask;
pub use execution::{BatchExecution, Compute, InterruptSignal};
pub use stochastic::{StochasticComputeExt, StochasticInput, StochasticTask};
pub use stream::{
    MultiStreams, RandomVariable, RngBundle, SeedSource, SingleStream, StreamSeed, StreamSeeds,
};
pub use transform::{
    DependentInput, DependentStochasticInput, StochasticTransform, Transform, TransformExt,
};

#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub struct NoFunction;
