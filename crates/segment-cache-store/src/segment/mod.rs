//! Immutable segment files and the invariants needed to encode, identify, open,
//! and read them.

mod file;
mod fingerprint;
mod format;
mod geometry;
mod index;
mod io;
mod state;
mod writer;

pub(crate) use file::SegmentOpenOptions;
pub(crate) use fingerprint::SegmentFingerprint;
pub(crate) use geometry::SegmentGeometry;
pub(crate) use state::Segment;
pub(crate) use writer::SegmentWriter;
