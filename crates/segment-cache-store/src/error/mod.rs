//! Error aggregation at the public API boundary.
//!
//! Persistence codecs own precise catalog, format, and corruption errors. This
//! module owns caller-contract errors, the top-level [`Error`] returned by
//! public APIs, and every conversion into it.
//!
//! [`CatalogError`]: crate::CatalogError
//! [`FormatError`]: crate::FormatError
//! [`CorruptionError`]: crate::CorruptionError

mod persistence;

pub(crate) use persistence::SegmentWriteError;
pub use persistence::{CorruptionError, FormatError};

#[cfg(feature = "value-compression")]
use crate::block::CompressionPolicyError;
use crate::catalog::{
    CatalogError, CatalogMismatch, ManifestEncodeError, ManifestParseError, StoreFileParseError,
};

/// Top-level error returned by the segment cache store.
#[derive(thiserror::Error, Debug)]
#[non_exhaustive]
pub enum Error {
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Input(#[from] InputError),

    #[error(transparent)]
    Catalog(#[from] CatalogError),

    #[error(transparent)]
    Format(#[from] FormatError),

    #[error(transparent)]
    Corruption(#[from] CorruptionError),
}

impl From<OptionsError> for Error {
    fn from(error: OptionsError) -> Self {
        Self::Input(InputError::InvalidOptions(error))
    }
}

#[cfg(feature = "value-compression")]
impl From<CompressionPolicyError> for Error {
    fn from(error: CompressionPolicyError) -> Self {
        Self::Input(InputError::InvalidOptions(error.into()))
    }
}

impl From<CatalogMismatch> for Error {
    fn from(error: CatalogMismatch) -> Self {
        Self::Catalog(CatalogError::Mismatch(error))
    }
}

impl From<StoreFileParseError> for Error {
    fn from(error: StoreFileParseError) -> Self {
        Self::Catalog(CatalogError::StoreParse(error))
    }
}

impl From<ManifestParseError> for Error {
    fn from(error: ManifestParseError) -> Self {
        Self::Catalog(CatalogError::ManifestParse(error))
    }
}

impl From<ManifestEncodeError> for Error {
    fn from(error: ManifestEncodeError) -> Self {
        Self::Catalog(CatalogError::ManifestEncode(error))
    }
}

impl From<SegmentWriteError> for Error {
    fn from(error: SegmentWriteError) -> Self {
        match error {
            SegmentWriteError::Io(error) => Self::Io(error),
            SegmentWriteError::Format(error) => Self::Format(error),
        }
    }
}

impl Error {
    /// Returns true when a read error should degrade to a cache miss.
    pub(crate) fn is_cache_miss_corruption(&self) -> bool {
        match self {
            Self::Corruption(CorruptionError::Block | CorruptionError::SegmentFormat) => true,
            Self::Io(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => true,
            _ => false,
        }
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Caller input or API contract violation.
#[derive(thiserror::Error, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum InputError {
    #[error("key length mismatch: expected {expected}, got {actual}")]
    WrongKeyLength { expected: usize, actual: usize },

    #[error("value length mismatch: expected fixed {expected}, got {actual}")]
    WrongValueLength { expected: u32, actual: usize },

    #[error("value exceeds the {max}-byte implementation limit: got {actual}")]
    ValueTooLarge { max: usize, actual: usize },

    #[error(transparent)]
    InvalidOptions(#[from] OptionsError),

    #[error("ordered lookup input is not sorted")]
    UnsortedLookupKeys,

    #[error("duplicate key found inside one commit batch")]
    DuplicateKeyInBatch,

    #[error("key already exists with different value bytes")]
    KeyConflict,

    #[error("store already exists")]
    StoreAlreadyExists,

    #[error("another writer holds the store lock")]
    WriterLocked,

    #[error("commit attempted on a read-only store handle")]
    ReadOnlyStore,

    #[error("source store metadata does not match destination store")]
    SourceMetadataMismatch,

    #[error("source store key length mismatch: expected {expected}, got {actual}")]
    SourceKeyLengthMismatch { expected: usize, actual: usize },

    #[error("source store value layout does not match destination store")]
    SourceValueLayoutMismatch,
}

/// Invalid creation or commit options.
#[derive(thiserror::Error, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum OptionsError {
    #[error("key_len must be greater than zero")]
    KeyLenZero,

    #[error("key_len exceeds the implementation limit")]
    KeyLenTooLarge,

    #[error("fixed value length exceeds the implementation limit")]
    FixedValueLenTooLarge,

    #[cfg(feature = "value-compression")]
    #[error(transparent)]
    CompressionPolicy(#[from] CompressionPolicyError),

    #[error("writable stores must keep block checksum verification enabled")]
    WritableStoreRequiresBlockChecksumVerification,
}

#[cfg(test)]
mod tests {
    mod cache_miss {
        use crate::{CorruptionError, Error, InputError};

        #[test]
        fn classification_is_narrow() {
            assert!(Error::Corruption(CorruptionError::Block).is_cache_miss_corruption());
            assert!(Error::Corruption(CorruptionError::SegmentFormat).is_cache_miss_corruption());
            assert!(
                !Error::Corruption(CorruptionError::DuplicateVisibleKey).is_cache_miss_corruption()
            );
            assert!(Error::Io(std::io::ErrorKind::UnexpectedEof.into()).is_cache_miss_corruption());
            assert!(
                !Error::Input(InputError::WrongKeyLength {
                    expected: 16,
                    actual: 4,
                })
                .is_cache_miss_corruption()
            );
        }
    }
}
