//! Pure on-disk format: encoding, decoding, and validation of every persisted
//! byte layout.
//!
//! This layer is self-contained: it depends only on `std` and the checksum
//! crate — no filesystem access, no other crate modules. Encoders write into
//! byte buffers or generic [`std::io::Write`] sinks; decoders read from byte
//! slices. Functions return the precise format-owned error types from
//! [`error`] (decoding fails as corruption, encoding fails on format limits);
//! aggregation into the public [`crate::Error`] happens at the API boundary in
//! `crate::error`. All file access, atomic publication, locking, and garbage
//! collection live in [`crate::engine`].
//!
//! Layout of this tree mirrors the design document:
//!
//! - [`store_file`]: the `STORE` descriptor (line-oriented text)
//! - [`manifest`]: the binary `MANIFEST` snapshot
//! - [`segment`]: segment header, footer, sparse block index, and writer
//! - [`block`]: data-block encoding, layout math, and decoding
//! - [`entry`], [`value`], [`metadata`]: shared logical-data vocabulary
//! - [`error`]: the format-owned error vocabulary

pub(crate) mod binary;
pub(crate) mod block;
pub(crate) mod entry;
mod error;
pub(crate) mod manifest;
mod metadata;
pub(crate) mod segment;
pub(crate) mod store_file;
mod value;

pub(crate) use error::SegmentWriteError;
pub use error::{CatalogError, CatalogMismatch, CorruptionError, FormatError};
pub use metadata::StoreMetadata;
pub use value::ValueLayout;

/// Length of the byte prefix shared by `left` and `right`.
pub(crate) fn common_prefix_len(left: &[u8], right: &[u8]) -> usize {
    left.iter()
        .zip(right)
        .take_while(|(left, right)| left == right)
        .count()
}

/// Converts a length or offset into the persisted `u32`, reporting the named
/// quantity on overflow.
pub(crate) fn format_u32(value: usize, quantity: &'static str) -> Result<u32, FormatError> {
    u32::try_from(value).map_err(|_| FormatError::limit(quantity))
}
