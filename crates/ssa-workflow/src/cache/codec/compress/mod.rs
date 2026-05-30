//! Compressed codec adapters layered on top of another [`CodecEngine`].

use core::num::NonZeroUsize;

use super::{CloneFresh, CodecEngine, Error as CodecError, SkipReason, ValueFormat};

mod algorithm;
mod frame;
mod policy;

pub use algorithm::Compress;
#[cfg(feature = "lz4")]
pub use algorithm::lz4::Lz4;
#[cfg(feature = "zstd")]
pub use algorithm::zstd::Zstd;
pub use frame::Error as CompressError;
pub use policy::{CompressPolicy, CompressionAction, DefaultCompressPolicy};

use self::frame::CompressFrame;

/// Extension trait for building compressed codec adapters from an existing codec value.
pub trait CompressCodecExt: CloneFresh + Sized {
    /// Wrap this codec in a compressed codec builder using `compressor`.
    fn compress<C>(self, compressor: C) -> CompressedCodecBuilder<Self, C> {
        CompressedCodecBuilder::new(self, compressor)
    }
}

impl<E> CompressCodecExt for E where E: CloneFresh {}

/// Builder returned by [`CompressCodecExt::compress`].
pub struct CompressedCodecBuilder<E, C, P = DefaultCompressPolicy> {
    inner: E,
    compressor: C,
    policy: P,
    max_len: usize,
}

impl<E, C> CompressedCodecBuilder<E, C, DefaultCompressPolicy> {
    fn new(inner: E, compressor: C) -> Self {
        Self {
            inner,
            compressor,
            policy: DefaultCompressPolicy::default(),
            max_len: 0,
        }
    }

    /// Set the smallest serialized payload that should attempt compression.
    pub fn with_min_try_compress_len(mut self, min_try_compress_len: usize) -> Self {
        self.policy = self.policy.with_min_try_compress_len(min_try_compress_len);
        self
    }

    /// Set the minimum saved ratio required to keep the compressed frame.
    pub fn with_min_saved_ratio(mut self, numerator: usize, denominator: usize) -> Self {
        self.policy = self.policy.with_min_saved_ratio(numerator, denominator);
        self
    }
}

impl<E, C, P> CompressedCodecBuilder<E, C, P> {
    /// Replace the compression policy.
    pub fn with_policy<P2>(self, policy: P2) -> CompressedCodecBuilder<E, C, P2> {
        let Self {
            inner,
            compressor,
            max_len,
            ..
        } = self;
        CompressedCodecBuilder {
            inner,
            compressor,
            policy,
            max_len,
        }
    }

    /// Set a size limit applied at both encode time and decode time.
    ///
    /// Passing `0` removes the limit.
    pub fn with_max_len(mut self, max_len: usize) -> Self {
        self.max_len = max_len;
        self
    }

    /// Build the compressed codec.
    pub fn build(self) -> CompressedCodec<E, C, P> {
        CompressedCodec::from_raw_parts(self.inner, self.compressor, self.policy, self.max_len)
    }
}

/// Codec adapter that stores values in a framed compressed format.
///
/// The inner codec serializes `T`; this adapter then stores either the raw serialized bytes or a
/// compressed frame, depending on the configured [`CompressPolicy`]. On reads, it recovers the
/// serialized bytes and passes them back to the inner codec.
///
/// Use [`CompressCodecExt::compress`] for the ergonomic builder API. Use
/// [`CompressedCodecBuilder::with_max_len`] to reject oversized payloads during encode and decode;
/// passing `0` disables that limit.
pub struct CompressedCodec<E, C, P = DefaultCompressPolicy> {
    inner: E,
    frame: CompressFrame<C>,
    policy: P,
    max_encode_len: Option<NonZeroUsize>,
}

impl<E, C, P> CompressedCodec<E, C, P> {
    /// Create a compressed codec from already-configured parts.
    ///
    /// `max_len` is applied at encode time to the serialized payload and at decode time to
    /// compressed payloads. Passing `0` disables the size limit.
    pub fn from_raw_parts(inner: E, compressor: C, policy: P, max_len: usize) -> Self {
        let max_len = NonZeroUsize::new(max_len);
        Self {
            inner,
            frame: CompressFrame::new(compressor).with_max_decode_len(max_len),
            policy,
            max_encode_len: max_len,
        }
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

impl<T, E, C, P> CodecEngine<T> for CompressedCodec<E, C, P>
where
    E: CodecEngine<T>,
    C: Compress + CloneFresh,
    P: CompressPolicy + Clone,
{
    const VALUE_FORMAT: ValueFormat = ValueFormat::concat(&E::VALUE_FORMAT, C::FORMAT_ID);

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
                if raw_size > u32::MAX as usize {
                    return Err(SkipReason::EncodedValueTooLarge {
                        encoded_len: raw_size,
                        max_len: u32::MAX as usize,
                    });
                }
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

    impl crate::cache::codec::CloneFresh for PassthroughBytesEngine {
        fn clone_fresh(&self) -> Self {
            Self::default()
        }
    }

    impl CodecEngine<Vec<u8>> for PassthroughBytesEngine {
        const VALUE_FORMAT: ValueFormat = ValueFormat::new("test-passthrough-bytes-v1");

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

    impl crate::cache::codec::CloneFresh for SizedBytesEngine {
        fn clone_fresh(&self) -> Self {
            Self::default()
        }
    }

    impl CodecEngine<usize> for SizedBytesEngine {
        const VALUE_FORMAT: ValueFormat = ValueFormat::new("test-sized-bytes-v1");

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

    use super::{
        super::{
            CodecEngine, SkipReason,
            fixtures::{Error as FixtureError, FixtureEngine},
        },
        fixtures::*,
        *,
    };

    #[derive(Default, Copy, Clone)]
    struct TestCompress;

    impl crate::cache::codec::CloneFresh for TestCompress {
        fn clone_fresh(&self) -> Self {
            *self
        }
    }

    impl algorithm::sealed::Sealed for TestCompress {}

    impl Compress for TestCompress {
        const ALGORITHM_ID: u8 = 15;
        const FORMAT_ID: &'static str = "test-compress-v1";

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

    impl crate::cache::codec::CloneFresh for RejectingEncodeEngine {
        fn clone_fresh(&self) -> Self {
            Self
        }
    }

    impl CodecEngine<Vec<u8>> for RejectingEncodeEngine {
        const VALUE_FORMAT: ValueFormat = ValueFormat::new("test-rejecting-encode-bytes-v1");

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

    impl crate::cache::codec::CloneFresh for RejectingDecodeEngine {
        fn clone_fresh(&self) -> Self {
            Self::default()
        }
    }

    impl CodecEngine<Vec<u8>> for RejectingDecodeEngine {
        const VALUE_FORMAT: ValueFormat = ValueFormat::new("test-rejecting-decode-bytes-v1");

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
        let mut engine = PassthroughBytesEngine::default()
            .compress(TestCompress)
            .build();

        let encoded = engine.encode(&value).unwrap().to_vec();
        let decoded = engine.decode(&encoded)?;

        assert_eq!(decoded, value);
        Ok(())
    }

    #[test]
    fn small_value_stays_uncompressed() -> crate::Result<()> {
        let value = vec![7u8; 1024];
        let mut engine = PassthroughBytesEngine::default()
            .compress(TestCompress)
            .build();

        let encoded = engine.encode(&value).unwrap().to_vec();
        let decoded = engine.decode(&encoded)?;

        assert_raw_header(&encoded);
        assert_eq!(decoded, value);
        Ok(())
    }

    #[test]
    fn incompressible_data_falls_back_to_raw() -> crate::Result<()> {
        let value: Vec<u8> = (0..(96 * 1024)).map(|i| (i % 251) as u8).collect();
        let mut engine = PassthroughBytesEngine::default()
            .compress(TestCompress)
            .build();

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
        let mut engine = SizedBytesEngine::default()
            .compress(TestCompress)
            .with_max_len(max_encode_len)
            .build();

        assert!(matches!(
            engine.encode(&oversize_len),
            Err(SkipReason::EncodedValueTooLarge { .. })
        ));
    }

    #[test]
    fn codec_propagates_inner_encode_skip_reason() {
        let value = vec![1u8; 128];
        let mut engine = RejectingEncodeEngine.compress(TestCompress).build();

        assert!(matches!(
            engine.encode(&value),
            Err(SkipReason::EncodedValueTooLarge { .. })
        ));
    }

    #[test]
    fn runtime_policy_instance_can_compress_small_values() -> crate::Result<()> {
        let value = vec![b'a'; 1024];
        let mut engine = PassthroughBytesEngine::default()
            .compress(TestCompress)
            .with_policy(AggressivePolicy)
            .build();

        let encoded = engine.encode(&value).unwrap().to_vec();
        let decoded = engine.decode(&encoded)?;

        assert_eq!(encoded[0] & 0b1111_0000, 0b0001_0000);
        assert_eq!(encoded[0] & 0b0000_1111, TestCompress::ALGORITHM_ID);
        assert_eq!(decoded, value);
        Ok(())
    }

    #[cfg(feature = "lz4")]
    #[test]
    fn lz4_frames_record_the_lz4_algorithm_id() -> crate::Result<()> {
        let value = "a".repeat(96 * 1024);
        let mut engine = FixtureEngine::default().compress(Lz4).build();

        let encoded = engine.encode(&value).unwrap().to_vec();
        let decoded: String = engine.decode(&encoded)?;

        assert!(!encoded.is_empty());
        assert_eq!(encoded[0] & 0b1111_0000, 0b0001_0000);
        assert_eq!(encoded[0] & 0b0000_1111, Lz4::ALGORITHM_ID);
        assert_eq!(decoded, value);
        Ok(())
    }

    #[test]
    fn codec_supports_raw_parts_constructor() -> crate::Result<()> {
        let value = vec![b'a'; 1024];
        let mut engine = CompressedCodec::from_raw_parts(
            PassthroughBytesEngine::default(),
            TestCompress,
            DefaultCompressPolicy::default(),
            0,
        );

        let encoded = engine.encode(&value).unwrap().to_vec();
        let decoded = engine.decode(&encoded)?;

        assert_eq!(decoded, value);
        Ok(())
    }

    #[test]
    fn codec_maps_frame_decode_errors() {
        let value = vec![b'a'; 96 * 1024];
        let mut engine = PassthroughBytesEngine::default()
            .compress(TestCompress)
            .with_policy(AggressivePolicy)
            .build();

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
        let mut engine = RejectingDecodeEngine::default()
            .compress(TestCompress)
            .build();

        let encoded = engine.encode(&value).unwrap().to_vec();

        assert!(matches!(
            engine.decode(&encoded),
            Err(CodecError::Fixture(_))
        ));
    }

    #[test]
    fn from_raw_parts_is_the_low_level_constructor() -> crate::Result<()> {
        let value = vec![b'a'; 96 * 1024];
        let mut engine = CompressedCodec::from_raw_parts(
            PassthroughBytesEngine::default(),
            TestCompress,
            AggressivePolicy,
            128 * 1024,
        );

        let encoded = engine.encode(&value).unwrap().to_vec();
        let decoded = engine.decode(&encoded)?;

        assert_eq!(decoded, value);
        Ok(())
    }

    #[test]
    fn from_raw_parts_max_len_rejects_oversized_payload() {
        let value = vec![b'a'; 128];
        let mut engine = CompressedCodec::from_raw_parts(
            PassthroughBytesEngine::default(),
            TestCompress,
            DefaultCompressPolicy::default(),
            1,
        );

        assert!(matches!(
            engine.encode(&value),
            Err(SkipReason::EncodedValueTooLarge {
                encoded_len: 128,
                max_len: 1
            })
        ));
    }

    #[test]
    fn codec_ext_builder_builds_compressed_codec() -> crate::Result<()> {
        let value = vec![b'a'; 96 * 1024];
        let mut engine = PassthroughBytesEngine::default()
            .compress(TestCompress)
            .with_policy(AggressivePolicy)
            .with_max_len(128 * 1024)
            .build();

        let encoded = engine.encode(&value).unwrap().to_vec();
        let decoded = engine.decode(&encoded)?;

        assert_eq!(decoded, value);
        Ok(())
    }

    #[test]
    fn codec_ext_builder_configures_default_policy() -> crate::Result<()> {
        let value = vec![b'a'; 8 * 1024];
        let mut engine = PassthroughBytesEngine::default()
            .compress(TestCompress)
            .with_min_try_compress_len(1)
            .with_min_saved_ratio(1, 20)
            .build();

        let encoded = engine.encode(&value).unwrap().to_vec();
        let decoded = engine.decode(&encoded)?;

        assert_eq!(decoded, value);
        Ok(())
    }

    #[test]
    fn codec_ext_builder_max_len_rejects_oversized_payload() {
        let value = vec![b'a'; 128];
        let mut engine = PassthroughBytesEngine::default()
            .compress(TestCompress)
            .with_max_len(1)
            .build();

        assert!(matches!(
            engine.encode(&value),
            Err(SkipReason::EncodedValueTooLarge {
                encoded_len: 128,
                max_len: 1
            })
        ));
    }

    #[test]
    fn clone_fresh_codec_encodes_and_decodes_independently() -> crate::Result<()> {
        let value = vec![b'a'; 96 * 1024];
        let original = PassthroughBytesEngine::default()
            .compress(TestCompress)
            .with_policy(AggressivePolicy)
            .build();
        let mut forked = original.clone_fresh();

        let encoded = forked.encode(&value).unwrap().to_vec();
        let decoded = forked.decode(&encoded)?;

        assert_eq!(decoded, value);
        Ok(())
    }

    #[test]
    fn clone_fresh_codec_is_independent_from_original() -> crate::Result<()> {
        let value_a = vec![b'a'; 96 * 1024];
        let value_b = vec![b'b'; 96 * 1024];
        let original = PassthroughBytesEngine::default()
            .compress(TestCompress)
            .with_policy(AggressivePolicy)
            .build();
        let mut forked = original.clone_fresh();
        let mut original = original;

        let encoded_a = original.encode(&value_a).unwrap().to_vec();
        let encoded_b = forked.encode(&value_b).unwrap().to_vec();

        assert_eq!(original.decode(&encoded_a)?, value_a);
        assert_eq!(forked.decode(&encoded_b)?, value_b);
        Ok(())
    }
}
