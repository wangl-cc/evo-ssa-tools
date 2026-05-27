//! Concrete compute node implementations.
//!
//! # TODO: reduce structural duplication
//!
//! The four node types — [`DeterministicTask`], [`StochasticTask`], [`Transform`], and
//! `StochasticTransform` — share a common structure that differs along two orthogonal axes:
//!
//! - **Root vs Dependent**: whether the node has an upstream [`Compute`] source.
//! - **Deterministic vs Stochastic**: whether the node requires an RNG stream.
//!
//! Currently each combination is a hand-written type with its own builder, `Clone` impl,
//! and [`Compute::execute_with_encoded_input`] body. This is readable and debug-friendly
//! but creates ~300 lines of near-duplicate code.
//!
//! When a fifth variant is needed (e.g. streaming, batched, or traced nodes), consolidate
//! the shared boilerplate with a `macro_rules!` generator. A trait-based unification is
//! impractical until Rust stabilises `impl Trait` in trait method signatures (tracking
//! [RPITIT](https://rust-lang.github.io/impl-trait-initiative/RFCs/rpitit.html)).

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
