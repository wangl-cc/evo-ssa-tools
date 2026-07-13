//! Errors owned by persistent catalog identity and visibility invariants.

use super::{ManifestEncodeError, ManifestParseError, StoreFileParseError};

/// Persistent `STORE`/`MANIFEST` catalog error.
///
/// Catalog errors reject an open or commit outright; they are never degraded
/// to cache misses.
#[derive(thiserror::Error, Debug, Eq, PartialEq)]
#[non_exhaustive]
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
#[non_exhaustive]
pub enum CatalogMismatch {
    #[error("missing STORE file")]
    MissingStore,

    #[error("missing MANIFEST file")]
    MissingManifest,

    #[error("metadata does not match")]
    Metadata,

    #[error("STORE key length is zero")]
    StoreKeyLenZero,

    #[error("STORE key length exceeds the implementation limit")]
    StoreKeyLenTooLarge,

    #[error("STORE fixed value length exceeds the implementation limit")]
    StoreValueLenTooLarge,

    #[error("STORE block checksum algorithm is unsupported: format id {format_id}")]
    UnsupportedBlockChecksum { format_id: u8 },

    #[error("STORE value payload compression is unsupported: format id {format_id}")]
    UnsupportedValuePayloadCompression { format_id: u8 },

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

    #[error("MANIFEST patch entries are not sorted by key and segment id")]
    SegmentPatchOrder,

    #[error("MANIFEST normalization component contains more than one patch segment")]
    MultiplePatchesInComponent,

    #[error("MANIFEST segment key length does not match STORE")]
    SegmentKeyLen,

    #[error("MANIFEST segment range is invalid")]
    SegmentKeyRange,

    #[error("segment id space is exhausted")]
    SegmentIdExhausted,
}
