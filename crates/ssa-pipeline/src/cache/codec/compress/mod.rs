//! Compressed codec adapters layered on top of another [`CodecEngine`].

use core::num::NonZeroUsize;

use super::{CodecEngine, Error as CodecError, SkipReason};
use crate::cache::CloneFresh;

pub mod policy;
#[cfg(feature = "lz4")]
pub use algorithm::Lz4;
#[cfg(feature = "zstd")]
pub use algorithm::Zstd;

pub use self::{
    algorithm::Compress,
    frame::Error as CompressError,
    policy::{CompressPolicy, CompressionAction, DefaultCompressPolicy},
};

mod algorithm;
mod frame;

use self::frame::CompressFrame;

/// Compression wrapper engine over a base serialization engine.
///
/// Each worker gets its own independent engine instance with a fresh inner engine and
/// scratch buffers; the compression policy is cloned rather than shared.
///
/// Payloads larger than `u32::MAX` bytes are outside the supported design
/// envelope of the framed compressed format. If compression is attempted for
/// such a payload, encoding may panic while constructing the compressed frame
/// header.
///
/// Encoding path:
///
/// - Serialize `T` with the inner engine `E`.
/// - Optionally reject oversized serialized payloads with [`CompressedCodec::with_max_len`] before
///   building a cache frame.
/// - Ask policy `P` whether the serialized form should stay raw or attempt compression with `C`.
/// - If compression is attempted, ask `P` whether to keep the compressed frame or fall back to raw.
///
/// Decoding path:
///
/// - Inspect the frame header.
/// - Either borrow the raw payload directly or decompress it into scratch space.
/// - Pass the recovered serialized bytes back to `E::decode`.
/// - Optionally reject oversized compressed payloads with [`CompressedCodec::with_max_len`] before
///   allocating decode scratch space.
pub struct CompressedCodec<E, C, P = DefaultCompressPolicy> {
    inner: E,
    frame: CompressFrame<C>,
    policy: P,
    max_encode_len: Option<NonZeroUsize>,
}

impl<E, C, P> CompressedCodec<E, C, P> {
    /// Create a compressed codec from an already-configured inner engine, compressor, and policy.
    pub fn from_parts(inner: E, compressor: C, policy: P) -> Self {
        Self {
            inner,
            frame: CompressFrame::new(compressor),
            policy,
            max_encode_len: None,
        }
    }

    /// Replace the compressor while keeping the inner engine and policy.
    pub fn with_compressor(mut self, compressor: C) -> Self {
        self.frame = self.frame.with_compressor(compressor);
        self
    }

    /// Replace the compression policy while keeping the inner engine and compressor.
    pub fn with_policy<P2>(self, policy: P2) -> CompressedCodec<E, C, P2> {
        let Self {
            inner,
            frame,
            max_encode_len,
            ..
        } = self;
        CompressedCodec {
            inner,
            frame,
            policy,
            max_encode_len,
        }
    }

    /// Set a size limit applied at both encode time (serialized payload) and decode time
    /// (compressed payload).
    ///
    /// Passing `0` removes the limit.
    pub fn with_max_len(mut self, max_len: usize) -> Self {
        let limit = NonZeroUsize::new(max_len);
        self.max_encode_len = limit;
        self.frame = self.frame.with_max_decode_len(limit);
        self
    }
}

impl<E, C> CompressedCodec<E, C, DefaultCompressPolicy> {
    /// Create a compressed codec from an already-configured inner engine.
    pub fn new(inner: E) -> Self
    where
        C: Default,
    {
        Self::from_parts(inner, C::default(), DefaultCompressPolicy::default())
    }
}

impl<E: CloneFresh, C: CloneFresh, P: Clone> CloneFresh for CompressedCodec<E, C, P> {
    fn clone_fresh(&self) -> Self {
        Self {
            inner: self.inner.clone_fresh(),
            frame: self.frame.clone_fresh(),
            policy: self.policy.clone(),
            max_encode_len: self.max_encode_len,
        }
    }
}

impl<E: Default, C: Default, P: Default> Default for CompressedCodec<E, C, P> {
    fn default() -> Self {
        Self::from_parts(E::default(), C::default(), P::default())
    }
}

impl<T, E, C, P> CodecEngine<T> for CompressedCodec<E, C, P>
where
    E: CodecEngine<T>,
    C: Compress,
    P: CompressPolicy,
{
    fn encode(&mut self, value: &T) -> Result<&[u8], SkipReason> {
        let Self {
            inner,
            frame,
            policy,
            max_encode_len,
        } = self;
        let raw = inner.encode(value)?;
        let raw_size = raw.len();
        if let Some(limit) = max_encode_len
            && raw_size > limit.get()
        {
            return Err(SkipReason::EncodedValueTooLarge {
                encoded_len: raw_size,
                max_len: limit.get(),
            });
        }
        let raw_frame_size = raw_size + frame::RAW_FRAME_EXTRA_LEN;
        match policy.before_compress(raw_size) {
            CompressionAction::Raw => {
                frame.encode_raw(raw);
                Ok(frame.get_frame(raw_frame_size))
            }
            CompressionAction::Compress => {
                let compressed_frame_size = frame.encode_compressed(raw);
                let action = policy.after_compress(raw_frame_size, compressed_frame_size);
                match action {
                    CompressionAction::Compress => Ok(frame.get_frame(compressed_frame_size)),
                    CompressionAction::Raw => {
                        frame.encode_raw(raw);
                        Ok(frame.get_frame(raw_frame_size))
                    }
                }
            }
        }
    }

    fn decode(&mut self, bytes: &[u8]) -> Result<T, CodecError> {
        let raw = self.frame.decompress(bytes)?;
        self.inner.decode(raw)
    }
}

/// Shared test fixtures for `compress` and its submodules.
#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
pub(crate) mod fixtures {
    use super::*;

    /// Minimal pass-through engine for testing [`CompressedCodec`] independent of bitcode.
    #[derive(Debug, Default)]
    pub struct PassthroughBytesEngine {
        buffer: Vec<u8>,
    }

    impl crate::cache::CloneFresh for PassthroughBytesEngine {
        fn clone_fresh(&self) -> Self {
            Self::default()
        }
    }

    impl CodecEngine<Vec<u8>> for PassthroughBytesEngine {
        fn encode(&mut self, value: &Vec<u8>) -> std::result::Result<&[u8], SkipReason> {
            self.buffer.clear();
            self.buffer.extend_from_slice(value);
            Ok(&self.buffer)
        }

        fn decode(&mut self, bytes: &[u8]) -> Result<Vec<u8>, CodecError> {
            Ok(bytes.to_vec())
        }
    }

    /// Test engine that materializes a zero-filled raw payload of the requested length.
    #[derive(Debug, Default)]
    pub struct SizedBytesEngine {
        buffer: Vec<u8>,
    }

    impl CodecEngine<usize> for SizedBytesEngine {
        fn encode(&mut self, value: &usize) -> std::result::Result<&[u8], SkipReason> {
            self.buffer.resize(*value, 0);
            Ok(&self.buffer)
        }

        fn decode(&mut self, bytes: &[u8]) -> Result<usize, CodecError> {
            Ok(bytes.len())
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use fixtures::{PassthroughBytesEngine, SizedBytesEngine};

    use super::*;
    use crate::cache::codec::{CodecEngine, SkipReason, fixtures::Error as FixtureError};

    #[derive(Default, Copy, Clone)]
    struct TestCompress;

    impl crate::cache::CloneFresh for TestCompress {
        fn clone_fresh(&self) -> Self {
            *self
        }
    }

    impl Compress for TestCompress {
        const ALGORITHM_ID: u8 = 15;

        fn max_output_size(&self, input_len: usize) -> usize {
            input_len.max(1)
        }

        unsafe fn compress_into_unchecked(&mut self, input: &[u8], output: &mut [u8]) -> usize {
            if input.is_empty() {
                return 0;
            }

            if input.iter().all(|byte| *byte == input[0]) {
                output[0] = input[0];
                1
            } else {
                output[..input.len()].copy_from_slice(input);
                input.len()
            }
        }

        unsafe fn decompress_into_unchecked(
            &mut self,
            input: &[u8],
            output: &mut [u8],
        ) -> Result<usize, frame::Error> {
            match input.len() {
                0 => Ok(0),
                1 => {
                    output.fill(input[0]);
                    Ok(output.len())
                }
                len if len == output.len() => {
                    output.copy_from_slice(input);
                    Ok(len)
                }
                _ => Err(frame::Error::TruncatedInput),
            }
        }
    }

    #[derive(Default)]
    struct RejectingEncodeEngine;

    impl CodecEngine<Vec<u8>> for RejectingEncodeEngine {
        fn encode(&mut self, value: &Vec<u8>) -> Result<&[u8], SkipReason> {
            Err(SkipReason::EncodedValueTooLarge {
                encoded_len: value.len(),
                max_len: 0,
            })
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

    #[derive(Default, Clone)]
    struct AggressivePolicy;

    impl CompressPolicy for AggressivePolicy {
        fn before_compress(&self, _: usize) -> policy::CompressionAction {
            policy::CompressionAction::Compress
        }

        fn after_compress(&self, _: usize, _: usize) -> policy::CompressionAction {
            policy::CompressionAction::Compress
        }
    }

    fn assert_raw_header(encoded: &[u8]) {
        assert!(!encoded.is_empty());
        assert_eq!(encoded[0] & 0b1111_0000, 0b0001_0000);
        assert_eq!(encoded[0] & 0b0000_1111, 0);
    }

    #[test]
    fn codec_roundtrip_uses_inner_engine() -> crate::Result<()> {
        let value = vec![b'a'; 96 * 1024];
        let mut engine: CompressedCodec<PassthroughBytesEngine, TestCompress> =
            CompressedCodec::new(PassthroughBytesEngine::default());

        let encoded = engine.encode(&value).unwrap().to_vec();
        let decoded = engine.decode(&encoded)?;

        assert_eq!(decoded, value);
        Ok(())
    }

    #[test]
    fn small_value_stays_uncompressed() -> crate::Result<()> {
        let value = vec![7u8; 1024];
        let mut engine: CompressedCodec<PassthroughBytesEngine, TestCompress> =
            CompressedCodec::new(PassthroughBytesEngine::default());

        let encoded = engine.encode(&value).unwrap().to_vec();
        let decoded = engine.decode(&encoded)?;

        assert_raw_header(&encoded);
        assert_eq!(decoded, value);
        Ok(())
    }

    #[test]
    fn incompressible_data_falls_back_to_raw() -> crate::Result<()> {
        let value: Vec<u8> = (0..(96 * 1024)).map(|i| (i % 251) as u8).collect();
        let mut engine: CompressedCodec<PassthroughBytesEngine, TestCompress> =
            CompressedCodec::new(PassthroughBytesEngine::default());

        let encoded = engine.encode(&value).unwrap().to_vec();
        let decoded = engine.decode(&encoded)?;

        assert_raw_header(&encoded);
        assert_eq!(decoded, value);
        Ok(())
    }

    #[test]
    fn oversize_raw_payload_skips_cache_encoding() {
        let max_encode_len = 64 * 1024 * 1024;
        let oversize_len = max_encode_len + 1;
        let mut engine: CompressedCodec<SizedBytesEngine, TestCompress> =
            CompressedCodec::new(SizedBytesEngine::default()).with_max_len(max_encode_len);

        assert!(matches!(
            engine.encode(&oversize_len),
            Err(SkipReason::EncodedValueTooLarge { .. })
        ));
    }

    #[test]
    fn codec_propagates_inner_encode_skip_reason() {
        let value = vec![1u8; 128];
        let mut engine: CompressedCodec<RejectingEncodeEngine, TestCompress> =
            CompressedCodec::new(RejectingEncodeEngine);

        assert!(matches!(
            engine.encode(&value),
            Err(SkipReason::EncodedValueTooLarge { .. })
        ));
    }

    #[test]
    fn runtime_policy_instance_can_compress_small_values() -> crate::Result<()> {
        let value = vec![b'a'; 1024];
        let mut engine: CompressedCodec<PassthroughBytesEngine, TestCompress, AggressivePolicy> =
            CompressedCodec::new(PassthroughBytesEngine::default()).with_policy(AggressivePolicy);

        let encoded = engine.encode(&value).unwrap().to_vec();
        let decoded = engine.decode(&encoded)?;

        assert_eq!(encoded[0] & 0b1111_0000, 0b0001_0000);
        assert_eq!(encoded[0] & 0b0000_1111, TestCompress::ALGORITHM_ID);
        assert_eq!(decoded, value);
        Ok(())
    }

    #[test]
    fn codec_supports_custom_compressor_instances() -> crate::Result<()> {
        let value = vec![b'a'; 1024];
        let mut engine = CompressedCodec::from_parts(
            PassthroughBytesEngine::default(),
            TestCompress,
            DefaultCompressPolicy::default(),
        );

        let encoded = engine.encode(&value).unwrap().to_vec();
        let decoded = engine.decode(&encoded)?;

        assert_eq!(decoded, value);
        Ok(())
    }

    #[test]
    fn codec_maps_frame_decode_errors() {
        let value = vec![b'a'; 96 * 1024];
        let mut engine: CompressedCodec<PassthroughBytesEngine, TestCompress, AggressivePolicy> =
            CompressedCodec::new(PassthroughBytesEngine::default()).with_policy(AggressivePolicy);

        let mut encoded = engine.encode(&value).unwrap().to_vec();
        let last = encoded.last_mut().expect("compressed frame is non-empty");
        *last ^= 0xFF;

        assert!(matches!(
            engine.decode(&encoded),
            Err(CodecError::Compress(_))
        ));
    }

    #[test]
    fn codec_propagates_inner_decode_errors() {
        let value = vec![7u8; 1024];
        let mut engine: CompressedCodec<RejectingDecodeEngine, TestCompress> =
            CompressedCodec::new(RejectingDecodeEngine::default());

        let encoded = engine.encode(&value).unwrap().to_vec();

        assert!(matches!(
            engine.decode(&encoded),
            Err(CodecError::Fixture(_))
        ));
    }

    #[test]
    fn codec_builder_methods_allow_replacing_components_and_removing_limits() -> crate::Result<()> {
        let value = vec![b'a'; 96 * 1024];
        let mut engine: CompressedCodec<PassthroughBytesEngine, TestCompress, AggressivePolicy> =
            CompressedCodec::new(PassthroughBytesEngine::default())
                .with_compressor(TestCompress)
                .with_max_len(1)
                .with_max_len(0)
                .with_policy(AggressivePolicy);

        let encoded = engine.encode(&value).unwrap().to_vec();
        let decoded = engine.decode(&encoded)?;

        assert_eq!(decoded, value);
        Ok(())
    }

    #[test]
    fn clone_fresh_codec_encodes_and_decodes_independently() -> crate::Result<()> {
        use crate::cache::CloneFresh;
        let value = vec![b'a'; 96 * 1024];
        let original: CompressedCodec<PassthroughBytesEngine, TestCompress, AggressivePolicy> =
            CompressedCodec::new(PassthroughBytesEngine::default()).with_policy(AggressivePolicy);
        let mut forked = original.clone_fresh();

        let encoded = forked.encode(&value).unwrap().to_vec();
        let decoded = forked.decode(&encoded)?;

        assert_eq!(decoded, value);
        Ok(())
    }

    #[test]
    fn clone_fresh_codec_is_independent_from_original() -> crate::Result<()> {
        use crate::cache::CloneFresh;
        let value_a = vec![b'a'; 96 * 1024];
        let value_b = vec![b'b'; 96 * 1024];
        let original: CompressedCodec<PassthroughBytesEngine, TestCompress, AggressivePolicy> =
            CompressedCodec::new(PassthroughBytesEngine::default()).with_policy(AggressivePolicy);
        let mut forked = original.clone_fresh();
        let mut original = original;

        let encoded_a = original.encode(&value_a).unwrap().to_vec();
        let encoded_b = forked.encode(&value_b).unwrap().to_vec();

        assert_eq!(original.decode(&encoded_a)?, value_a);
        assert_eq!(forked.decode(&encoded_b)?, value_b);
        Ok(())
    }
}
