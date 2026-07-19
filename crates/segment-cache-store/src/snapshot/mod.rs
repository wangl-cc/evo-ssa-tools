//! Read operations over one immutable catalog-selected segment snapshot.
//!
//! Readers own references to an immutable visible segment snapshot, so they
//! remain valid across commits.

mod ordered;
mod point;
mod range;

use crate::{Error, segment::SegmentGeometry};

#[derive(Clone, Copy)]
pub(crate) enum CorruptionHandling {
    AsCacheMiss,
    Strict,
}

impl CorruptionHandling {
    pub(crate) fn degrades(self, error: &Error) -> bool {
        matches!(self, Self::AsCacheMiss) && error.is_cache_miss_corruption()
    }

    #[cfg(feature = "value-compression")]
    pub(crate) fn degrades_corruption(self) -> bool {
        matches!(self, Self::AsCacheMiss)
    }
}

#[derive(Clone, Copy)]
pub(crate) struct LookupReadOptions {
    pub(crate) geometry: SegmentGeometry,
    pub(crate) verify_block_checksums: bool,
    pub(crate) corruption_handling: CorruptionHandling,
}

pub use ordered::OrderedLookup;
pub(crate) use point::SegmentSetReader;
pub use range::RangeCursor;
