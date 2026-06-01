//! Error types and cache-specific error classification.

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("I/O error")]
    Io(#[from] std::io::Error),

    #[error("manifest parse error: {reason}")]
    ManifestParse { reason: String },

    #[error("key length mismatch: expected {expected}, got {actual}")]
    WrongKeyLength { expected: usize, actual: usize },

    #[error("value length mismatch: expected fixed {expected}, got {actual}")]
    WrongValueLength { expected: usize, actual: usize },

    #[error("invalid store option: {reason}")]
    InvalidOptions { reason: &'static str },

    #[error("ordered lookup input is not sorted")]
    UnsortedLookupKeys,

    #[error("duplicate key found inside one shard batch")]
    DuplicateKeyInBatch,

    #[error(
        "out-of-order append for shard {shard}: batch min key is not greater than last published max key"
    )]
    OutOfOrderAppend { shard: usize },

    #[error("store options do not match existing manifest: {reason}")]
    ManifestMismatch { reason: &'static str },

    #[error("store uses unsupported format version {version}")]
    UnsupportedFormatVersion { version: u32 },

    #[error("corrupt or malformed block")]
    CorruptBlock,
}

impl Error {
    /// Returns true when a read error should degrade to a cache miss.
    pub(crate) fn is_cache_miss_corruption(&self) -> bool {
        match self {
            Self::CorruptBlock => true,
            Self::UnsupportedFormatVersion { .. } => true,
            Self::Io(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => true,
            _ => false,
        }
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
