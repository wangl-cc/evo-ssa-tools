//! Format-layer error vocabulary.
//!
//! These types are owned by the format layer: every constructor lives in
//! `format` (plus a few engine sites that enforce the same catalog
//! invariants). They say precisely how each codec can fail — decoding fails as
//! corruption, encoding fails on format limits, and catalog parse/encode errors
//! are owned by the catalog file modules themselves. Aggregation into the public
//! [`crate::Error`] happens in `crate::error`, the API boundary.

use crate::format::{
    manifest::{ManifestEncodeError, ManifestParseError},
    store_file::StoreFileParseError,
};

/// Persistent `STORE`/`MANIFEST` catalog error.
///
/// Catalog errors reject an open or commit outright; they are never degraded
/// to cache misses.
#[derive(thiserror::Error, Debug, Eq, PartialEq)]
pub enum CatalogError {
    /// `STORE` bytes are malformed.
    #[error(transparent)]
    StoreParse(#[from] StoreFileParseError),

    /// `MANIFEST` bytes are malformed or corrupt.
    #[error(transparent)]
    ManifestParse(#[from] ManifestParseError),

    /// Encoding would exceed the v1 binary `MANIFEST` capacity.
    #[error(transparent)]
    ManifestEncode(#[from] ManifestEncodeError),

    /// Parsed catalog content does not match store invariants or caller metadata.
    #[error(transparent)]
    Mismatch(#[from] CatalogMismatch),

    /// A catalog file declares an unsupported format version.
    #[error("unsupported {file} version {version}")]
    UnsupportedVersion { file: &'static str, version: u32 },
}

/// Parsed catalog content does not match store invariants or caller metadata.
#[derive(thiserror::Error, Debug, Eq, PartialEq)]
pub enum CatalogMismatch {
    #[error("missing STORE file")]
    MissingStore,

    #[error("missing MANIFEST file")]
    MissingManifest,

    #[error("metadata does not match")]
    Metadata,

    #[error("STORE key length is zero")]
    StoreKeyLenZero,

    #[error("STORE key length exceeds u32")]
    StoreKeyLenTooLarge,

    #[error("STORE block checksum algorithm is unsupported: format id {format_id}")]
    UnsupportedBlockChecksum { format_id: u32 },

    #[error("STORE value payload compression is unsupported: format id {format_id}")]
    UnsupportedValuePayloadCompression { format_id: u32 },

    #[error("MANIFEST key length does not match STORE")]
    ManifestKeyLen,

    #[error("duplicate segment id in MANIFEST")]
    DuplicateSegmentId,

    #[error("MANIFEST next_segment_id reuses an existing id")]
    NextSegmentId,

    #[error("MANIFEST segment ranges overlap")]
    SegmentOverlap,

    #[error("MANIFEST patch entries must follow main entries")]
    SegmentTierOrder,

    #[error("MANIFEST segment key length does not match STORE")]
    SegmentKeyLen,

    #[error("MANIFEST segment range is invalid")]
    SegmentKeyRange,

    #[error("segment id space is exhausted")]
    SegmentIdExhausted,
}

/// Encoding would exceed the v1 on-disk format capacity.
///
/// Every overflow is the same caller decision — the data being written does
/// not fit the format's `u32`-based envelope — so this is one type carrying
/// the overflowing quantity as a diagnostic, not a variant per field.
#[derive(thiserror::Error, Clone, Debug, Eq, PartialEq)]
#[error("{quantity} exceeds the v1 on-disk format limit")]
pub struct FormatError {
    quantity: &'static str,
}

impl FormatError {
    pub(crate) fn limit(quantity: &'static str) -> Self {
        Self { quantity }
    }

    /// The encoded quantity that overflowed, e.g. `"block length"`.
    #[must_use]
    pub fn quantity(&self) -> &'static str {
        self.quantity
    }
}

/// Published cache data is malformed or corrupt and should degrade to miss.
#[derive(thiserror::Error, Debug, Eq, PartialEq)]
pub enum CorruptionError {
    #[error("corrupt or malformed block")]
    Block,

    #[error("corrupt, malformed, or unsupported segment format")]
    SegmentFormat,
}

/// Failure while streaming one segment into a sink: either the sink failed or
/// the data exceeds format limits.
///
/// Crate-internal: the public API surfaces both cases through [`crate::Error`].
#[derive(thiserror::Error, Debug)]
pub(crate) enum SegmentWriteError {
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Format(#[from] FormatError),
}
