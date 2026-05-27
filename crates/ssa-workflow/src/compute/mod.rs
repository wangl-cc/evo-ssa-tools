//! Concrete compute node implementations.

pub mod deterministic;
pub mod execution;
pub mod stochastic;
pub mod stream;
pub mod transform;

pub use deterministic::DeterministicTask;
pub use execution::{BatchExecution, Compute, InterruptSignal};
pub use stochastic::{StochasticInput, StochasticTask};
pub use stream::{NamedStreams, RandomVariable, RngStreams, SingleStream, StreamSeed, StreamSeeds};
pub use transform::{DependentInput, DependentStochasticInput, Transform, TransformExt};

#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub struct NoFunction;
