//! Block checksum identity and optional digest implementation.

#[cfg(feature = "block-checksum")]
mod digest;

/// Built-in checksum used for data-block metadata and payload sections.
///
/// The variant's format id is persisted in `STORE` and segment headers.
/// Changing a checksum's bytes, seed, output width, or collision semantics
/// requires assigning a new variant and id.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum BlockChecksumKind {
    /// Store no block checksum bytes.
    ///
    /// This is useful for controlled benchmarks or disposable cache data. It
    /// does not provide corruption detection.
    None,

    /// CRC32C block checksum.
    #[cfg(feature = "checksum-crc32c")]
    Crc32c,

    /// Rapidhash V3 64-bit block checksum.
    #[cfg(feature = "checksum-rapidhash")]
    RapidHashV3_64,
}

impl BlockChecksumKind {
    /// Resolves a persisted checksum id to a built-in checksum.
    pub(crate) fn from_format_id(format_id: u8) -> Option<Self> {
        match format_id {
            0 => Some(Self::None),
            #[cfg(feature = "checksum-crc32c")]
            1 => Some(Self::Crc32c),
            #[cfg(feature = "checksum-rapidhash")]
            2 => Some(Self::RapidHashV3_64),
            _ => None,
        }
    }

    /// Human-readable checksum name used in diagnostics and debug output.
    pub const fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            #[cfg(feature = "checksum-crc32c")]
            Self::Crc32c => "crc32c",
            #[cfg(feature = "checksum-rapidhash")]
            Self::RapidHashV3_64 => "rapidhash-v3-64",
        }
    }

    /// Stable persisted checksum identifier.
    pub(crate) fn format_id(self) -> u8 {
        match self {
            Self::None => 0,
            #[cfg(feature = "checksum-crc32c")]
            Self::Crc32c => 1,
            #[cfg(feature = "checksum-rapidhash")]
            Self::RapidHashV3_64 => 2,
        }
    }

    /// Persisted digest width in bytes.
    pub(crate) fn digest_len(self) -> usize {
        match self {
            Self::None => 0,
            #[cfg(feature = "checksum-crc32c")]
            Self::Crc32c => 4,
            #[cfg(feature = "checksum-rapidhash")]
            Self::RapidHashV3_64 => 8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_has_reserved_zero_identity() {
        let checksum = BlockChecksumKind::None;
        assert_eq!(checksum.format_id(), 0);
        assert_eq!(checksum.digest_len(), 0);
        assert_eq!(BlockChecksumKind::from_format_id(0), Some(checksum));
    }

    #[cfg(not(feature = "checksum-crc32c"))]
    #[test]
    fn crc32c_id_is_unavailable_without_backend() {
        assert_eq!(BlockChecksumKind::from_format_id(1), None);
    }

    #[cfg(not(feature = "checksum-rapidhash"))]
    #[test]
    fn rapidhash_id_is_unavailable_without_backend() {
        assert_eq!(BlockChecksumKind::from_format_id(2), None);
    }

    #[cfg(feature = "checksum-crc32c")]
    #[test]
    fn crc32c_has_stable_identity_and_width() {
        let checksum = BlockChecksumKind::Crc32c;
        assert_eq!(checksum.format_id(), 1);
        assert_eq!(checksum.digest_len(), 4);
        assert_eq!(BlockChecksumKind::from_format_id(1), Some(checksum));
    }

    #[cfg(feature = "checksum-rapidhash")]
    #[test]
    fn rapidhash_has_stable_identity_and_width() {
        let checksum = BlockChecksumKind::RapidHashV3_64;
        assert_eq!(checksum.format_id(), 2);
        assert_eq!(checksum.digest_len(), 8);
        assert_eq!(BlockChecksumKind::from_format_id(2), Some(checksum));
    }
}
