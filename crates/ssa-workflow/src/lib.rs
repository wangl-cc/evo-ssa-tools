#![doc = include_str!("../README.md")]
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

pub mod cache;
pub mod compute;
pub mod error;
pub mod identity;
pub mod prelude;

pub use compute::{BatchExecution, BatchProgress, BatchProgressSnapshot, Compute, InterruptSignal};
pub use error::{Error, Result};
