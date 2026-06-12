//! Error aggregation at the public API boundary.
//!
//! The format layer owns precise error types ([`CatalogError`],
//! [`FormatError`], [`CorruptionError`] — declared in `format::error`) and its
//! functions return them directly. This module owns the caller-contract enums,
//! the top-level [`Error`] that public APIs return, and every conversion into
//! it.
//!
//! [`CatalogError`]: crate::CatalogError
//! [`FormatError`]: crate::FormatError
//! [`CorruptionError`]: crate::CorruptionError

use crate::format::{
    CatalogError, CatalogMismatch, CorruptionError, FormatError, SegmentWriteError,
};

/// Top-level error returned by the segment cache store.
#[derive(thiserror::Error, Debug)]
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

impl From<CatalogMismatch> for Error {
    fn from(error: CatalogMismatch) -> Self {
        Self::Catalog(CatalogError::Mismatch(error))
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
            Self::Corruption(_) => true,
            Self::Io(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => true,
            _ => false,
        }
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Caller input or API contract violation.
#[derive(thiserror::Error, Debug, Eq, PartialEq)]
pub enum InputError {
    #[error("key length mismatch: expected {expected}, got {actual}")]
    WrongKeyLength { expected: usize, actual: usize },

    #[error("value length mismatch: expected fixed {expected}, got {actual}")]
    WrongValueLength { expected: u32, actual: usize },

    #[error(transparent)]
    InvalidOptions(#[from] OptionsError),

    #[error("ordered lookup input is not sorted")]
    UnsortedLookupKeys,

    #[error("duplicate key found inside one commit batch")]
    DuplicateKeyInBatch,

    #[error("new segment key range overlaps already-published data")]
    SegmentOverlap,

    #[error("store already exists")]
    StoreAlreadyExists,

    #[error("another writer holds the store lock")]
    WriterLocked,

    #[error("commit attempted on a read-only store handle")]
    ReadOnlyStore,
}

/// Invalid creation or commit options.
#[derive(thiserror::Error, Debug, Eq, PartialEq)]
pub enum OptionsError {
    #[error("key_len must be greater than zero")]
    KeyLenZero,

    #[error("key_len must fit in u32")]
    KeyLenTooLarge,

    #[error("target_block_size must fit in u32")]
    TargetBlockSizeTooLarge,

    #[error("flush_threshold_records must be greater than zero")]
    FlushThresholdRecordsZero,

    #[error("flush_threshold_bytes must be greater than zero")]
    FlushThresholdBytesZero,
}

#[cfg(test)]
mod tests {
    mod cache_miss {
        use crate::{CorruptionError, Error, InputError};

        #[test]
        fn classification_is_narrow() {
            assert!(Error::Corruption(CorruptionError::Block).is_cache_miss_corruption());
            assert!(Error::Corruption(CorruptionError::SegmentFormat).is_cache_miss_corruption());
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
