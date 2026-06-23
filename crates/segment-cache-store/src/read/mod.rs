//! Ordered read paths: lookup sessions and range cursors.
//!
//! Readers operate on an immutable segment snapshot taken from the engine's
//! runtime state, so they remain valid across commits.

pub(crate) mod cursor;
pub(crate) mod lookup;

pub use cursor::RangeCursor;
pub use lookup::OrderedLookup;
