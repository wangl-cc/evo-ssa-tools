//! Block checksum identity and digest implementation.

mod digest;

const DIGEST_LEN: usize = 16;

/// The checksum used for data-block metadata and payload sections.
///
/// The variant's format id is persisted in `STORE` and segment headers. The
/// checksum is a fixed store-wide algorithm rather than a per-store choice:
/// every segment carries the same id, and unknown ids are rejected at open.
/// Changing the checksum's bytes, seed, output width, or collision semantics
/// requires assigning a new variant and id.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum BlockChecksumKind {
    /// Unseeded XXH3 128-bit block checksum.
    Xxh3_128,
}

/// The block checksum used for every segment in every store.
pub(crate) const BLOCK_CHECKSUM: BlockChecksumKind = BlockChecksumKind::Xxh3_128;

impl BlockChecksumKind {
    /// Resolves a persisted checksum id to a built-in checksum.
    pub(crate) fn from_format_id(format_id: u8) -> Option<Self> {
        match format_id {
            1 => Some(Self::Xxh3_128),
            _ => None,
        }
    }

    /// Human-readable checksum name used in diagnostics and debug output.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Xxh3_128 => "xxh3-128",
        }
    }

    /// Stable persisted checksum identifier.
    pub(crate) fn format_id(self) -> u8 {
        match self {
            Self::Xxh3_128 => 1,
        }
    }

    /// Persisted digest width in bytes.
    pub(crate) fn digest_len(self) -> usize {
        match self {
            Self::Xxh3_128 => DIGEST_LEN,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xxh3_128_has_stable_identity_and_width() {
        let checksum = BlockChecksumKind::Xxh3_128;
        assert_eq!(checksum.format_id(), 1);
        assert_eq!(checksum.digest_len(), 16);
        assert_eq!(BlockChecksumKind::from_format_id(1), Some(checksum));
    }

    #[test]
    fn unassigned_ids_are_rejected() {
        assert_eq!(BlockChecksumKind::from_format_id(0), None);
        assert_eq!(BlockChecksumKind::from_format_id(2), None);
        assert_eq!(BlockChecksumKind::from_format_id(3), None);
        assert_eq!(BlockChecksumKind::from_format_id(255), None);
    }
}
