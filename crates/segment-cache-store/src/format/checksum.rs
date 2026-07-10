//! Built-in block checksum implementations and their persisted identities.
//!
//! The checksum used for data blocks is part of the store format: a block
//! written with one checksum implementation cannot be safely decoded by a
//! different implementation. The crate intentionally exposes a small enum of
//! built-in choices instead of a third-party extension trait, keeping the disk
//! identity space owned by this crate.

/// Maximum supported block-checksum digest width in bytes.
///
/// This keeps verification allocation-free on the hot path while leaving room
/// for wider checksums such as 128-bit hashes.
pub(crate) const MAX_BLOCK_CHECKSUM_LEN: usize = 32;

/// Built-in checksum used for data-block metadata and payload sections.
///
/// The variant's `format_id` is persisted in `STORE` and segment headers.
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
    /// Default block checksum used by [`crate::CreateOptions::new`].
    #[cfg(feature = "checksum-rapidhash")]
    pub const DEFAULT: Self = Self::RapidHashV3_64;

    /// Resolves a persisted checksum id to a built-in checksum.
    pub(crate) const fn from_format_id(format_id: u32) -> Option<Self> {
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
    pub(crate) const fn format_id(self) -> u32 {
        match self {
            Self::None => 0,
            #[cfg(feature = "checksum-crc32c")]
            Self::Crc32c => 1,
            #[cfg(feature = "checksum-rapidhash")]
            Self::RapidHashV3_64 => 2,
        }
    }

    /// Persisted digest width in bytes.
    pub(crate) const fn digest_len(self) -> usize {
        match self {
            Self::None => 0,
            #[cfg(feature = "checksum-crc32c")]
            Self::Crc32c => 4,
            #[cfg(feature = "checksum-rapidhash")]
            Self::RapidHashV3_64 => 8,
        }
    }

    pub(crate) fn digest_into(self, bytes: &[u8], out: &mut [u8]) {
        debug_assert_eq!(out.len(), self.digest_len());
        #[cfg(not(any(feature = "checksum-crc32c", feature = "checksum-rapidhash")))]
        let _ = bytes;
        match self {
            Self::None => debug_assert!(out.is_empty()),
            #[cfg(feature = "checksum-crc32c")]
            Self::Crc32c => out.copy_from_slice(&crc32c::crc32c(bytes).to_le_bytes()),
            #[cfg(feature = "checksum-rapidhash")]
            Self::RapidHashV3_64 => {
                out.copy_from_slice(&rapidhash::v3::rapidhash_v3(bytes).to_le_bytes());
            }
        }
    }

    /// Returns whether `stored` is the checksum digest for `bytes`.
    pub(crate) fn verify(self, bytes: &[u8], stored: &[u8]) -> bool {
        let digest_len = self.digest_len();
        if stored.len() != digest_len {
            return false;
        }
        if digest_len == 0 {
            return true;
        }
        let mut expected = [0u8; MAX_BLOCK_CHECKSUM_LEN];
        self.digest_into(bytes, &mut expected[..digest_len]);
        expected[..digest_len] == *stored
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod kind {
        use super::*;

        #[test]
        fn no_checksum_has_reserved_zero_identity() {
            let checksum = BlockChecksumKind::None;
            assert_eq!(checksum.format_id(), 0);
            assert_eq!(checksum.digest_len(), 0);
            assert_eq!(BlockChecksumKind::from_format_id(0), Some(checksum));
            assert!(checksum.verify(b"changed bytes", &[]));
        }

        #[cfg(feature = "checksum-crc32c")]
        #[test]
        fn crc32c_detects_changed_bytes() {
            let checksum = BlockChecksumKind::Crc32c;
            let mut digest = vec![0u8; checksum.digest_len()];
            checksum.digest_into(b"hello", &mut digest);

            assert_eq!(checksum.format_id(), 1);
            assert_eq!(BlockChecksumKind::from_format_id(1), Some(checksum));
            assert!(checksum.verify(b"hello", &digest));
            assert!(!checksum.verify(b"jello", &digest));
        }

        #[cfg(feature = "checksum-rapidhash")]
        #[test]
        fn rapidhash_uses_eight_byte_digest() {
            let checksum = BlockChecksumKind::RapidHashV3_64;
            let mut digest = vec![0u8; checksum.digest_len()];
            checksum.digest_into(b"hello", &mut digest);

            assert_eq!(checksum.format_id(), 2);
            assert_eq!(digest.len(), 8);
            assert_eq!(BlockChecksumKind::from_format_id(2), Some(checksum));
            assert!(checksum.verify(b"hello", &digest));
            assert!(!checksum.verify(b"jello", &digest));
        }
    }
}
