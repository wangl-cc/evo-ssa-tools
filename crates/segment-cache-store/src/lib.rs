#![doc = include_str!("../README.md")]

mod error;
mod io;
mod manifest;
mod options;
mod read;
mod segment;
mod state;
mod store;
mod write;

pub use error::{Error, Result};
pub use options::{StoreOptions, ValueLayout};
pub use read::{cursor::RangeCursor, lookup::OrderedLookup};
pub use store::Store;
pub use write::batch::{CommitStats, WriteBatch};
