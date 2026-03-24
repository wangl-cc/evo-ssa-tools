//! Checked codec adapters layered on top of another [`CodecEngine`].

use crc32c::crc32c;

use super::{CodecEngine, Error as CodecError, SkipReason};

/// Bytes used by the trailing CRC32C checksum.
pub const CHECKSUM_BYTES: usize = core::mem::size_of::<u32>();

/// Decode-time failures specific to the checked framing layer.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Truncated input")]
    TruncatedInput,

    #[error("Checked frame checksum mismatch")]
    ChecksumMismatch,
}

impl Error {
    pub(crate) const fn is_cache_corruption(&self) -> bool {
        matches!(self, Self::TruncatedInput | Self::ChecksumMismatch)
    }
}

/// CRC32C wrapper engine over a base serialization engine.
///
/// Encoding path:
///
/// - Serialize `T` with the inner engine `E`.
/// - Append a trailing CRC32C checksum over the serialized payload.
///
/// Decoding path:
///
/// - Split the raw payload from the trailing checksum.
/// - Verify the checksum before handing bytes back to `E::decode`.
pub struct CheckedCodec<E> {
    inner: E,
    scratch: Vec<u8>,
}

impl<E> CheckedCodec<E> {
    /// Create a checked codec from an already-configured inner engine.
    pub fn new(inner: E) -> Self {
        Self {
            inner,
            scratch: Vec::new(),
        }
    }
}

impl<E: crate::cache::Fork> crate::cache::Fork for CheckedCodec<E> {
    fn fork(&self) -> Self {
        Self {
            inner: self.inner.fork(),
            scratch: Vec::new(),
        }
    }
}

impl<E: Default> Default for CheckedCodec<E> {
    fn default() -> Self {
        Self::new(E::default())
    }
}

impl<T, E> CodecEngine<T> for CheckedCodec<E>
where
    E: CodecEngine<T>,
{
    fn encode(&mut self, value: &T) -> Result<&[u8], SkipReason> {
        let raw = self.inner.encode(value)?;

        let total_len = raw.len() + CHECKSUM_BYTES;
        if self.scratch.len() < total_len {
            self.scratch.resize(total_len, 0);
        }
        self.scratch[..raw.len()].copy_from_slice(raw);
        let checksum = crc32c(raw).to_le_bytes();
        self.scratch[raw.len()..total_len].copy_from_slice(&checksum);
        Ok(&self.scratch[..total_len])
    }

    fn decode(&mut self, bytes: &[u8]) -> Result<T, CodecError> {
        if bytes.len() < CHECKSUM_BYTES {
            return Err(Error::TruncatedInput.into());
        }

        let (payload, checksum_bytes) = bytes.split_at(bytes.len() - CHECKSUM_BYTES);

        let stored_checksum = u32::from_le_bytes(
            checksum_bytes
                .try_into()
                .expect("split_at() ensures checksum is exactly 4 bytes"),
        );
        let actual_checksum = crc32c(payload);
        if actual_checksum != stored_checksum {
            return Err(Error::ChecksumMismatch.into());
        }

        self.inner.decode(payload)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::cache::codec::{CodecEngine, fixtures::Error as FixtureError};

    #[derive(Debug, Default)]
    struct PassthroughBytesEngine {
        buffer: Vec<u8>,
    }

    impl crate::cache::Fork for PassthroughBytesEngine {
        fn fork(&self) -> Self {
            Self::default()
        }
    }

    impl CodecEngine<Vec<u8>> for PassthroughBytesEngine {
        fn encode(&mut self, value: &Vec<u8>) -> Result<&[u8], SkipReason> {
            self.buffer.clear();
            self.buffer.extend_from_slice(value);
            Ok(&self.buffer)
        }

        fn decode(&mut self, bytes: &[u8]) -> Result<Vec<u8>, CodecError> {
            Ok(bytes.to_vec())
        }
    }

    #[derive(Default)]
    struct RejectingDecodeEngine {
        buffer: Vec<u8>,
    }

    impl CodecEngine<Vec<u8>> for RejectingDecodeEngine {
        fn encode(&mut self, value: &Vec<u8>) -> Result<&[u8], SkipReason> {
            self.buffer.clear();
            self.buffer.extend_from_slice(value);
            Ok(&self.buffer)
        }

        fn decode(&mut self, _: &[u8]) -> Result<Vec<u8>, CodecError> {
            Err(CodecError::from(FixtureError::Invalid(
                "inner decode rejected payload",
            )))
        }
    }

    fn checked_frame(payload: &[u8]) -> Vec<u8> {
        let mut bytes = payload.to_vec();
        bytes.extend_from_slice(&crc32c(payload).to_le_bytes());
        bytes
    }

    fn decode_checked_error(bytes: &[u8]) -> Error {
        let mut engine = CheckedCodec::new(PassthroughBytesEngine::default());
        match engine.decode(bytes) {
            Err(CodecError::Checked(err)) => err,
            Err(other) => panic!("expected checked error, got {other:?}"),
            Ok(_) => panic!("expected decode error"),
        }
    }

    #[test]
    fn roundtrip_preserves_payload() {
        let payload = b"checked codec payload".to_vec();
        let mut engine = CheckedCodec::new(PassthroughBytesEngine::default());

        let encoded = engine
            .encode(&payload)
            .expect("encoding should succeed")
            .to_vec();
        let decoded = engine.decode(&encoded).expect("decoding should succeed");

        assert_eq!(decoded, payload);
    }

    #[test]
    fn default_constructor_roundtrip() {
        let payload = b"default checked codec payload".to_vec();
        let mut engine = CheckedCodec::<PassthroughBytesEngine>::default();

        let encoded = engine
            .encode(&payload)
            .expect("encoding should succeed")
            .to_vec();
        let decoded = engine.decode(&encoded).expect("decoding should succeed");

        assert_eq!(decoded, payload);
    }

    #[test]
    fn roundtrip_preserves_empty_payload() {
        let payload = Vec::new();
        let mut engine = CheckedCodec::new(PassthroughBytesEngine::default());

        let encoded = engine
            .encode(&payload)
            .expect("encoding should succeed")
            .to_vec();

        assert_eq!(encoded.len(), CHECKSUM_BYTES);
        assert_eq!(
            engine.decode(&encoded).expect("decoding should succeed"),
            payload
        );
    }

    #[test]
    fn checksum_mismatch_detects_corruption() {
        let base = checked_frame(b"abcdef");

        for encoded in [
            {
                let mut encoded = base.clone();
                encoded[0] ^= 0x01;
                encoded
            },
            {
                let mut encoded = base.clone();
                let last = encoded.len() - 1;
                encoded[last] ^= 0x01;
                encoded
            },
        ] {
            assert!(matches!(
                decode_checked_error(&encoded),
                Error::ChecksumMismatch
            ));
        }
    }

    #[test]
    fn short_input_is_rejected() {
        assert!(matches!(
            decode_checked_error(&[1, 2, 3]),
            Error::TruncatedInput
        ));
    }

    #[test]
    fn corruption_classification_matches_policy() {
        assert!(Error::TruncatedInput.is_cache_corruption());
        assert!(Error::ChecksumMismatch.is_cache_corruption());
    }

    #[test]
    fn inner_decode_errors_propagate() {
        let payload = b"abc".to_vec();
        let mut encode_engine = CheckedCodec::new(PassthroughBytesEngine::default());
        let encoded = encode_engine
            .encode(&payload)
            .expect("encoding should succeed")
            .to_vec();

        let mut decode_engine = CheckedCodec::new(RejectingDecodeEngine::default());
        match decode_engine.decode(&encoded) {
            Err(CodecError::Fixture(FixtureError::Invalid(message))) => {
                assert_eq!(message, "inner decode rejected payload");
            }
            other => panic!("expected fixture error, got {other:?}"),
        }
    }

    #[test]
    fn fork_produces_independent_engine() {
        use crate::cache::Fork;
        let payload = b"fork test".to_vec();
        let original = CheckedCodec::new(PassthroughBytesEngine::default());
        let mut forked = original.fork();

        let encoded = forked
            .encode(&payload)
            .expect("encoding should succeed")
            .to_vec();
        let decoded = forked.decode(&encoded).expect("decoding should succeed");
        assert_eq!(decoded, payload);
    }
}
