#![doc = include_str!("../README.md")]
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

pub mod cache;
pub mod compute;
pub mod error;
pub mod identity;
pub mod prelude;

pub use compute::{BatchExecution, Compute, InterruptSignal};
pub use error::{Error, Result};
#[cfg(feature = "derive")]
pub use ssa_workflow_derive::CanonicalEncode;
