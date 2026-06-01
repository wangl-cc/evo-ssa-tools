#![doc = include_str!("../README.md")]

mod batch;
mod binary;
mod block;
mod cursor;
mod error;
mod format;
mod lookup;
mod manifest;
mod options;
mod state;
mod store;
#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests;

pub use batch::{CommitStats, WriteBatch};
pub use cursor::RangeCursor;
pub use error::{Error, Result};
pub use lookup::OrderedLookup;
pub use options::{StoreOptions, ValueLayout};
pub use store::Store;
