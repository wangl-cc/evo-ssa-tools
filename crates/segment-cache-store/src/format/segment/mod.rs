//! Segment file format: header, footer, sparse block index, and writer.
//!
//! A segment is an immutable sorted file. The fixed-size header identifies
//! format and store geometry; the variable-size footer at the end of the file
//! is the completion marker and owns the sparse block index. Reading segment
//! bytes from disk lives in the engine; this module only encodes and decodes
//! them.

mod footer;
mod header;
mod index;
mod writer;

pub(crate) use footer::SegmentFooter;
pub(crate) use header::SegmentHeader;
pub(crate) use index::BlockIndexEntry;
pub(crate) use writer::SegmentWriter;

pub(crate) const SEGMENT_FORMAT_VERSION: u32 = 1;
pub(crate) const SEGMENT_HEADER_LEN: usize = 24;
pub(crate) const SEGMENT_FOOTER_TRAILER_LEN: usize = 8;
pub(crate) const HEADER_MAGIC: &[u8; 4] = b"SCSG";
