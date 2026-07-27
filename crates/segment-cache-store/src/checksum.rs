//! Fixed checksum implementation for version-1 persisted bytes.
//!
//! Version 1 uses canonical big-endian, unseeded XXH3-128 digests for
//! MANIFEST, segment footer, and block sections. The algorithm is part of the
//! format rather than a runtime choice.

pub(crate) const DIGEST_LEN: usize = 16;

pub(crate) fn digest(bytes: &[u8]) -> [u8; DIGEST_LEN] {
    xxhash_rust::xxh3::xxh3_128(bytes).to_be_bytes()
}

pub(crate) fn verify(bytes: &[u8], expected_digest: &[u8]) -> bool {
    digest(bytes).as_slice() == expected_digest
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calculates_canonical_known_digests() {
        // Canonical vectors from the xxHash 0.8.3 reference `xxhsum -H2`.
        assert_eq!(digest(b""), [
            0x99, 0xaa, 0x06, 0xd3, 0x01, 0x47, 0x98, 0xd8, 0x60, 0x01, 0xc3, 0x24, 0x46, 0x8d,
            0x49, 0x7f,
        ]);
        assert_eq!(digest(b"hello"), [
            0xb5, 0xe9, 0xc1, 0xad, 0x07, 0x1b, 0x3e, 0x7f, 0xc7, 0x79, 0xcf, 0xaa, 0x5e, 0x52,
            0x38, 0x18,
        ]);
        assert_eq!(digest(&[0; 256]), [
            0xbb, 0x03, 0x9d, 0x71, 0xe9, 0x63, 0x0a, 0x91, 0xa6, 0x8d, 0xfb, 0xb1, 0xd7, 0x5c,
            0x6e, 0x8d,
        ]);
    }

    #[test]
    fn detects_changed_bytes() {
        let expected = digest(b"hello");
        assert!(verify(b"hello", &expected));
        assert!(!verify(b"jello", &expected));
    }
}
