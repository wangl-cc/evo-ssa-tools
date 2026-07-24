//! Block checksum calculation and verification.

use super::{BlockChecksumKind, DIGEST_LEN};

impl BlockChecksumKind {
    pub(crate) fn digest(self, bytes: &[u8]) -> [u8; DIGEST_LEN] {
        match self {
            Self::Xxh3_128 => xxhash_rust::xxh3::xxh3_128(bytes).to_be_bytes(),
        }
    }

    /// Returns whether `stored` is the checksum digest for `bytes`.
    pub(crate) fn verify(self, bytes: &[u8], expected_digest: &[u8]) -> bool {
        self.digest(bytes).as_slice() == expected_digest
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calculates_canonical_known_digests() {
        let checksum = BlockChecksumKind::Xxh3_128;
        // Canonical vectors from the xxHash 0.8.3 reference `xxhsum -H2`.
        assert_eq!(checksum.digest(b""), [
            0x99, 0xaa, 0x06, 0xd3, 0x01, 0x47, 0x98, 0xd8, 0x60, 0x01, 0xc3, 0x24, 0x46, 0x8d,
            0x49, 0x7f,
        ]);
        assert_eq!(checksum.digest(b"hello"), [
            0xb5, 0xe9, 0xc1, 0xad, 0x07, 0x1b, 0x3e, 0x7f, 0xc7, 0x79, 0xcf, 0xaa, 0x5e, 0x52,
            0x38, 0x18,
        ]);
        assert_eq!(checksum.digest(&[0; 256]), [
            0xbb, 0x03, 0x9d, 0x71, 0xe9, 0x63, 0x0a, 0x91, 0xa6, 0x8d, 0xfb, 0xb1, 0xd7, 0x5c,
            0x6e, 0x8d,
        ]);
    }

    #[test]
    fn detects_changed_bytes() {
        let checksum = BlockChecksumKind::Xxh3_128;
        let digest = checksum.digest(b"hello");
        assert!(checksum.verify(b"hello", &digest));
        assert!(!checksum.verify(b"jello", &digest));
    }
}
