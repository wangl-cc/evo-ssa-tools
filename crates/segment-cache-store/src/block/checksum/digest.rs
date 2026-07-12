//! Block checksum calculation and verification.

use super::BlockChecksumKind;

const MAX_DIGEST_LEN: usize = 32;

impl BlockChecksumKind {
    pub(crate) fn digest(self, _bytes: &[u8]) -> impl AsRef<[u8]> + use<> {
        match self {
            Self::None => Digest {
                bytes: [0; MAX_DIGEST_LEN],
                len: 0,
            },
            #[cfg(feature = "checksum-crc32c")]
            Self::Crc32c => {
                let mut bytes = [0; MAX_DIGEST_LEN];
                bytes[..4].copy_from_slice(&crc32c::crc32c(_bytes).to_le_bytes());
                Digest { bytes, len: 4 }
            }
            #[cfg(feature = "checksum-rapidhash")]
            Self::RapidHashV3_64 => {
                let mut bytes = [0; MAX_DIGEST_LEN];
                bytes[..8].copy_from_slice(&rapidhash::v3::rapidhash_v3(_bytes).to_le_bytes());
                Digest { bytes, len: 8 }
            }
        }
    }

    /// Returns whether `stored` is the checksum digest for `bytes`.
    pub(crate) fn verify(self, bytes: &[u8], expected_digest: &[u8]) -> bool {
        self.digest(bytes).as_ref() == expected_digest
    }
}

struct Digest {
    bytes: [u8; MAX_DIGEST_LEN],
    len: usize,
}

impl AsRef<[u8]> for Digest {
    fn as_ref(&self) -> &[u8] {
        &self.bytes[..self.len]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod none {
        use super::*;

        #[test]
        fn calculates_empty_digest() {
            assert_eq!(BlockChecksumKind::None.digest(b"hello").as_ref(), b"");
        }

        #[test]
        fn accepts_empty_digest() {
            assert!(BlockChecksumKind::None.verify(b"changed bytes", &[]));
        }
    }

    #[cfg(feature = "checksum-crc32c")]
    mod crc32c {
        use super::*;

        #[test]
        fn calculates_known_digest() {
            let checksum = BlockChecksumKind::Crc32c;
            let digest = checksum.digest(b"hello");
            assert_eq!(digest.as_ref(), &[0x4c, 0xbb, 0x71, 0x9a]);
        }

        #[test]
        fn detects_changed_bytes() {
            let checksum = BlockChecksumKind::Crc32c;
            let digest = checksum.digest(b"hello");
            assert!(checksum.verify(b"hello", digest.as_ref()));
            assert!(!checksum.verify(b"jello", digest.as_ref()));
        }
    }

    #[cfg(feature = "checksum-rapidhash")]
    mod rapidhash {
        use super::*;

        #[test]
        fn calculates_known_digest() {
            let checksum = BlockChecksumKind::RapidHashV3_64;
            let digest = checksum.digest(b"hello");
            assert_eq!(digest.as_ref(), &[
                0x46, 0x79, 0x5f, 0xb4, 0x51, 0x76, 0x2d, 0x2e
            ]);
        }

        #[test]
        fn detects_changed_bytes() {
            let checksum = BlockChecksumKind::RapidHashV3_64;
            let digest = checksum.digest(b"hello");
            assert!(checksum.verify(b"hello", digest.as_ref()));
            assert!(!checksum.verify(b"jello", digest.as_ref()));
        }
    }
}
