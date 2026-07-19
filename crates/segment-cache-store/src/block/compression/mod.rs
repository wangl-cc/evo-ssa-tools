//! Value-payload compression identity and optional write/read implementation.

#[cfg(feature = "value-compression")]
mod codec;
#[cfg(feature = "value-compression")]
mod frame;
#[cfg(feature = "value-compression")]
mod policy;

/// Store-wide value-payload compression kind.
///
/// The kind is persisted in `STORE` and every segment header. Each concrete
/// compression feature adds the matching persisted variant; `None` is always
/// available so no-feature builds can read and write raw payload stores.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ValuePayloadCompressionKind {
    /// Store value payload bytes exactly as provided.
    None,

    /// LZ4 block compression for value payloads.
    #[cfg(feature = "value-compression-lz4")]
    Lz4,

    /// Zstandard level-1 compression for value payloads.
    #[cfg(feature = "value-compression-zstd")]
    ZstdLevel1,
}

impl ValuePayloadCompressionKind {
    /// Default value-payload compression used by [`crate::CreateOptions::new`].
    pub const DEFAULT: Self = Self::None;

    /// Resolves a persisted compression id to a built-in kind.
    pub(crate) fn from_format_id(format_id: u8) -> Option<Self> {
        match format_id {
            0 => Some(Self::None),
            #[cfg(feature = "value-compression-lz4")]
            1 => Some(Self::Lz4),
            #[cfg(feature = "value-compression-zstd")]
            2 => Some(Self::ZstdLevel1),
            _ => None,
        }
    }

    /// Human-readable compression name used in diagnostics and debug output.
    pub const fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            #[cfg(feature = "value-compression-lz4")]
            Self::Lz4 => "lz4",
            #[cfg(feature = "value-compression-zstd")]
            Self::ZstdLevel1 => "zstd_level1",
        }
    }

    /// Stable persisted compression identifier.
    pub(crate) fn format_id(self) -> u8 {
        match self {
            Self::None => 0,
            #[cfg(feature = "value-compression-lz4")]
            Self::Lz4 => 1,
            #[cfg(feature = "value-compression-zstd")]
            Self::ZstdLevel1 => 2,
        }
    }
}

#[cfg(feature = "value-compression")]
pub(crate) use codec::{ValuePayloadDecoder, ValuePayloadEncoder};
#[cfg(feature = "value-compression")]
pub use policy::{CompressionPolicyError, ValuePayloadCompressionPolicy};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_has_reserved_zero_identity() {
        assert_eq!(ValuePayloadCompressionKind::None.format_id(), 0);
        assert_eq!(
            ValuePayloadCompressionKind::from_format_id(0),
            Some(ValuePayloadCompressionKind::None)
        );
    }

    #[cfg(not(feature = "value-compression-lz4"))]
    #[test]
    fn lz4_id_is_unavailable_without_backend() {
        assert_eq!(ValuePayloadCompressionKind::from_format_id(1), None);
    }

    #[cfg(not(feature = "value-compression-zstd"))]
    #[test]
    fn zstd_id_is_unavailable_without_backend() {
        assert_eq!(ValuePayloadCompressionKind::from_format_id(2), None);
    }
}
