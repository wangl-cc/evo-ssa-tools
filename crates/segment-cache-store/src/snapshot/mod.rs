//! Read operations over one immutable catalog-selected segment snapshot.
//!
//! Readers own references to an immutable visible segment snapshot, so they
//! remain valid across commits.

mod ordered;
mod point;
mod range;

use crate::segment::SegmentGeometry;

#[derive(Clone, Copy)]
pub(crate) struct LookupReadOptions {
    pub(crate) geometry: SegmentGeometry,
    pub(crate) verify_block_checksums: bool,
}

pub use ordered::OrderedLookup;
pub(crate) use point::SegmentSetReader;
pub use range::RangeCursor;
