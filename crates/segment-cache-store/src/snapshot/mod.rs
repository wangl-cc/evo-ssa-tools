//! Read operations over one immutable visible main/patch segment snapshot.
//!
//! Readers own references to an immutable visible segment snapshot, so they
//! remain valid across commits.

pub(crate) mod ordered;
pub(crate) mod point;
pub(crate) mod range;

use crate::schema::StoreGeometry;

#[derive(Clone, Copy)]
pub(crate) struct LookupReadOptions {
    pub(crate) geometry: StoreGeometry,
    pub(crate) verify_block_checksums: bool,
}

pub use ordered::OrderedLookup;
pub use range::RangeCursor;
