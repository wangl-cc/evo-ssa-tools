//! Immutable segment files and the invariants needed to encode, identify, open,
//! and read them.

mod content_id;
mod file;
mod format;
mod geometry;
mod index;
mod io;
mod state;
mod writer;

pub(crate) use content_id::{SEGMENT_CONTENT_ID_LEN, SegmentContentId};
pub(crate) use file::SegmentOpenOptions;
pub(crate) use geometry::SegmentGeometry;
pub(crate) use state::Segment;
pub(crate) use writer::SegmentWriter;
