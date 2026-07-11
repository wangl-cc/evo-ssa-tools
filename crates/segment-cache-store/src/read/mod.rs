//! Ordered read paths: lookup sessions and range cursors.
//!
//! Readers operate on an immutable segment snapshot taken from the engine's
//! runtime state, so they remain valid across commits.

pub(crate) mod cursor;
pub(crate) mod lookup;
pub(crate) mod point;

use crate::engine::runtime::StoreGeometry;

#[derive(Clone, Copy)]
pub(crate) struct LookupReadOptions {
    pub(crate) geometry: StoreGeometry,
    pub(crate) verify_block_checksums: bool,
}

pub use cursor::RangeCursor;
pub use lookup::OrderedLookup;
