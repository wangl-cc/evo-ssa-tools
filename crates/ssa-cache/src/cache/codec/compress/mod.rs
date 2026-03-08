//! Framing and compression adapters layered on top of another [`CodecEngine`].
//!
//! [`CompressedCodec`] serializes values with an inner engine `E`, then
//! optionally compresses those serialized bytes with `C`. The outer frame is
//! self-describing: the first byte identifies the wire-format version and
//! whether the payload is raw or compressed.
//!
//! Compression policy lives in this module rather than in [`Compress`]:
//! small values stay raw, large values are compressed only when that produces a
//! meaningful frame-size win, and oversized serialized payloads are skipped
//! from cache writes via [`SkipReason`].

use crc32c::crc32c;

use super::{CodecEngine, Error as CodecError, SkipReason};
use crate::Result;

/// Compression algorithm adapter.
///
/// Implementors provide a block-level compress/decompress pair used by [`CompressedCodec`].
///
/// This trait does not decide *when* compression should be attempted or skipped.
/// That policy lives in [`CompressedCodec`]. `Compress` only describes how to
/// transform one raw byte slice into one compressed byte slice and back.
pub trait Compress {
    /// Algorithm identifier stored in the low 4 bits of the frame header.
    ///
    /// Assigned ids in the current format:
    ///
    /// - `0`: raw/uncompressed payload (reserved by the frame format)
    /// - `1`: [`lz4::Lz4`] when the `lz4` feature is enabled
    /// - `2..=15`: currently unassigned
    const ALGORITHM_ID: u8;

    /// Return the maximum output size to compress `input_len` bytes.
    ///
    /// This is used to pre-allocate output buffers before compression.
    ///
    /// # Buffer contract
    ///
    /// This method returns the maximum output size, which is used to pre-allocate buffers.
    /// The actual compressed size may be smaller.
    fn max_output_size(input_len: usize) -> usize;

    /// Compress `input` into `output` and return the compressed length.
    ///
    /// # Buffer contract
    ///
    /// The buffer `output` **must** be at least `Self::max_output_size(input.len())` length.
    /// [`CompressedCodec`] upholds this invariant before every call;
    /// implementations may panic if it is violated.
    fn compress_into(input: &[u8], output: &mut [u8]) -> usize;
    /// Decompress `input` into `output` and return the decompressed length.
    ///
    /// # Buffer contract
    ///
    /// The buffer `output` **must** be large enough to hold the decompressed data.
    /// [`CompressedCodec`] upholds this invariant before every call;
    /// implementations may panic if it is violated.
    fn decompress_into(input: &[u8], output: &mut [u8]) -> Result<usize, Error>;
}

/// Decode-time failures specific to the framed compression layer.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Header mismatch")]
    HeaderMismatch,

    #[error("Version unsupported {0}")]
    UnsupportedVersion(u8),

    #[error("Truncated input")]
    TruncatedInput,

    #[error("Decompressed length mismatch")]
    DecompressedLengthMismatch,

    #[error("Compression algorithm mismatch")]
    CompressionAlgorithmMismatch,

    #[error("Compressed frame checksum mismatch")]
    ChecksumMismatch,

    #[error("Content too large")]
    ContentTooLarge,

    #[cfg(feature = "lz4")]
    #[error("Lz4 decompression error")]
    Lz4(#[from] lz4_flex::block::DecompressError),
}

#[cfg(feature = "lz4")]
pub mod lz4;

// ── Compressed codec ────────────────────────────────────────────────────────

/// One-byte frame header used by [`CompressedCodec`].
///
/// # Bit Layout
///
/// ```text
///   bit7..4  bit3..0
/// +---------+--------+
/// | version |  algo  |
/// +---------+--------+
/// ```
///
/// - `version`: format version (`0b0001` currently).
/// - `algo`: algorithm id (`0` for uncompressed payloads). Non-zero values represent compressed
///   payloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Header {
    value: u8,
}

impl Header {
    const ALGORITHM_MASK: u8 = 0b0000_1111;
    const ALGORITHM_RAW: u8 = 0;
    /// The raw header value for an uncompressed payload.
    const RAW: Self = Self::from_inner(Self::VERSION_ENCODED | Self::ALGORITHM_RAW);
    const VERSION_ENCODED: u8 = Self::VERSION_V1 << Self::VERSION_SHIFT;
    const VERSION_MASK: u8 = 0b1111_0000;
    const VERSION_SHIFT: u8 = 4;
    /// Current wire-format version.
    const VERSION_V1: u8 = 0b001;

    const fn new_compressed(algorithm_id: u8) -> Self {
        assert!(
            0 < algorithm_id && algorithm_id <= Self::ALGORITHM_MASK,
            "algorithm id out of range (must be 1..=15)",
        );
        let value = Self::VERSION_ENCODED | (algorithm_id & Self::ALGORITHM_MASK);
        Self { value }
    }

    const fn into_inner(self) -> u8 {
        self.value
    }

    const fn from_inner(value: u8) -> Self {
        Self { value }
    }

    const fn version(self) -> u8 {
        (self.value & Self::VERSION_MASK) >> Self::VERSION_SHIFT
    }

    const fn algorithm(self) -> u8 {
        self.value & Self::ALGORITHM_MASK
    }
}

/// Reusable framing buffer for [`CompressedCodec`].
///
/// The first byte of every frame is a [`Header`]; see [`Header`] for the
/// exact bit layout.
///
/// # Wire Format
///
/// Raw frame:
///
/// ```text
/// [header][raw_payload...][checksum_crc32c_u32_le]
/// ```
///
/// Raw-frame checksum coverage:
///
/// ```text
/// crc32c(header || raw_payload)
/// ```
///
/// Compressed frame:
///
/// ```text
/// [header][original_len_u32_le][compressed_payload...][checksum_crc32c_u32_le]
/// ```
///
/// Compressed-frame checksum coverage:
///
/// ```text
/// crc32c(header || original_len_u32_le || compressed_payload)
/// ```
///
/// `CompressFrame` owns the reusable scratch buffer used to build frames during
/// encode and to materialize raw serialized bytes during decode.
struct CompressFrame<C> {
    scratch: Vec<u8>,
    _compress: std::marker::PhantomData<C>,
}

impl<C> Default for CompressFrame<C> {
    fn default() -> Self {
        Self {
            scratch: Vec::new(),
            _compress: std::marker::PhantomData,
        }
    }
}

impl<C: Compress> CompressFrame<C> {
    /// The checksum stored in framed payloads.
    const CHECKSUM_BYTES: usize = core::mem::size_of::<u32>();
    /// `[header][orig_len_u32_le]`
    const COMPRESSED_PREFIX_LEN: usize = 1 + Self::ORIGINAL_LEN_BYTES;
    /// Only attempt compression for moderately large payloads (64 KiB)
    const COMPRESSION_THRESHOLD: usize = 64 * 1024;
    /// Cap untrusted declared output size before allocating decode scratch (64 MiB).
    const MAX_DECOMPRESSED_LEN: usize = 64 * 1024 * 1024;
    /// Require a meaningful frame-size win before keeping the compressed form (4 KiB).
    const MIN_COMPRESSION_SAVINGS: usize = 4 * 1024;
    /// The length prefix stored in compressed frames.
    const ORIGINAL_LEN_BYTES: usize = core::mem::size_of::<u32>();
    /// `[header]`
    const RAW_PREFIX_LEN: usize = 1;

    /// Encode an uncompressed frame as `[header][raw_payload...][checksum_crc32c_u32_le]`.
    fn encode_raw(&mut self, raw: &[u8]) -> &[u8] {
        let req_len = Self::RAW_PREFIX_LEN + raw.len() + Self::CHECKSUM_BYTES;
        let scratch = &mut self.scratch;
        if scratch.len() < req_len {
            scratch.resize(req_len, 0);
        }
        let header = Header::RAW;
        scratch[0] = header.into_inner();
        scratch[Self::RAW_PREFIX_LEN..Self::RAW_PREFIX_LEN + raw.len()].copy_from_slice(raw);
        let checksum = crc32c(&scratch[..Self::RAW_PREFIX_LEN + raw.len()]);
        scratch[Self::RAW_PREFIX_LEN + raw.len()..req_len].copy_from_slice(&checksum.to_le_bytes());
        &scratch[..req_len]
    }

    /// Encode either a raw frame or a compressed frame.
    ///
    /// Returns [`SkipReason::EncodedValueTooLarge`] when the serialized payload is
    /// too large to be accepted by the compressed-frame safety bound.
    fn compress(&mut self, raw: &[u8]) -> Result<&[u8], SkipReason> {
        if raw.len() > Self::MAX_DECOMPRESSED_LEN {
            return Err(SkipReason::EncodedValueTooLarge {
                encoded_len: raw.len(),
                max_len: Self::MAX_DECOMPRESSED_LEN,
            });
        }

        // If the input is not big enough to compress, return the raw data.
        if raw.len() < Self::COMPRESSION_THRESHOLD {
            return Ok(self.encode_raw(raw));
        }

        // Compress the raw data
        let max_size = C::max_output_size(raw.len());
        let compressed_frame_capacity =
            Self::COMPRESSED_PREFIX_LEN + max_size + Self::CHECKSUM_BYTES;
        if self.scratch.len() < compressed_frame_capacity {
            self.scratch.resize(compressed_frame_capacity, 0);
        }
        let compressed_len = C::compress_into(
            raw,
            &mut self.scratch[Self::COMPRESSED_PREFIX_LEN..Self::COMPRESSED_PREFIX_LEN + max_size],
        );
        let total_compressed_len =
            Self::COMPRESSED_PREFIX_LEN + compressed_len + Self::CHECKSUM_BYTES;

        // If compression doesn't save enough bytes, return the raw data instead.
        let total_raw_len = Self::RAW_PREFIX_LEN + raw.len() + Self::CHECKSUM_BYTES;
        let saved_bytes = total_raw_len.saturating_sub(total_compressed_len);
        if saved_bytes < Self::MIN_COMPRESSION_SAVINGS {
            return Ok(self.encode_raw(raw));
        }

        // Write the header into the scratch buffer.
        let header = const { Header::new_compressed(C::ALGORITHM_ID) };
        let original_len =
            u32::try_from(raw.len()).expect("compressed payload length is capped below u32::MAX");
        self.scratch[0] = header.into_inner();
        let original_len_bytes = original_len.to_le_bytes();
        self.scratch[1..Self::COMPRESSED_PREFIX_LEN].copy_from_slice(&original_len_bytes);
        let checksum = crc32c(&self.scratch[..Self::COMPRESSED_PREFIX_LEN + compressed_len]);
        self.scratch[Self::COMPRESSED_PREFIX_LEN + compressed_len..total_compressed_len]
            .copy_from_slice(&checksum.to_le_bytes());

        // Return the compressed data including the header.
        Ok(&self.scratch[..total_compressed_len])
    }

    /// Decode a framed payload into raw serialized bytes for the inner codec.
    ///
    /// Raw frames borrow directly from the input slice. Compressed frames
    /// decompress into the reusable scratch buffer owned by this frame.
    fn decompress<'a>(&'a mut self, bytes: &'a [u8]) -> Result<&'a [u8], Error> {
        let (header_byte, payload) = bytes.split_first().ok_or(Error::HeaderMismatch)?;
        let header = Header::from_inner(*header_byte);

        let version = header.version();
        if version != Header::VERSION_V1 {
            return Err(Error::UnsupportedVersion(version));
        }

        let scratch = &mut self.scratch;
        match header.algorithm() {
            Header::ALGORITHM_RAW => {
                if payload.len() < Self::CHECKSUM_BYTES {
                    return Err(Error::TruncatedInput);
                }
                let (raw_payload, checksum_bytes) =
                    payload.split_at(payload.len() - Self::CHECKSUM_BYTES);
                let stored_checksum = u32::from_le_bytes(
                    checksum_bytes
                        .try_into()
                        .expect("payload is checked to be at least CHECKSUM_BYTES"),
                );
                let actual_checksum = crc32c(&bytes[..bytes.len() - Self::CHECKSUM_BYTES]);
                if actual_checksum != stored_checksum {
                    return Err(Error::ChecksumMismatch);
                }
                Ok(raw_payload)
            }
            id if id == C::ALGORITHM_ID => {
                if payload.len() < Self::ORIGINAL_LEN_BYTES + Self::CHECKSUM_BYTES {
                    return Err(Error::TruncatedInput);
                }
                let (len_bytes, rest) = payload.split_at(Self::ORIGINAL_LEN_BYTES);
                let (compressed_payload, checksum_bytes) =
                    rest.split_at(rest.len() - Self::CHECKSUM_BYTES);
                let stored_checksum = u32::from_le_bytes(
                    checksum_bytes
                        .try_into()
                        .expect("payload is checked to be at least CHECKSUM_BYTES"),
                );
                let actual_checksum = crc32c(&bytes[..bytes.len() - Self::CHECKSUM_BYTES]);
                if actual_checksum != stored_checksum {
                    return Err(Error::ChecksumMismatch);
                }
                let original_len = u32::from_le_bytes(
                    len_bytes
                        .try_into()
                        .expect("payload is checked to be at least ORIGINAL_LEN_BYTES"),
                ) as usize;
                if original_len > Self::MAX_DECOMPRESSED_LEN {
                    return Err(Error::ContentTooLarge);
                }
                if scratch.len() < original_len {
                    scratch.resize(original_len, 0);
                }
                let decoded_len =
                    C::decompress_into(compressed_payload, &mut scratch[..original_len])?;
                if decoded_len != original_len {
                    return Err(Error::DecompressedLengthMismatch);
                }
                Ok(&scratch[..decoded_len])
            }
            _ => Err(Error::CompressionAlgorithmMismatch),
        }
    }
}

/// Compression wrapper engine over a base serialization engine.
///
/// Encoding path:
///
/// - Serialize `T` with the inner engine `E`.
/// - If the serialized form is large enough, attempt compression with `C`.
/// - Keep the compressed frame only if it clears the configured savings threshold.
/// - Return [`SkipReason`] if the serialized form exceeds the compressed-frame safety limit.
///
/// Decoding path:
///
/// - Inspect the frame header.
/// - Either borrow the raw payload directly or decompress it into scratch space.
/// - Pass the recovered serialized bytes back to `E::decode`.
pub struct CompressedCodec<E, C> {
    inner: E,
    frame: CompressFrame<C>,
}

impl<E: Default, C> Default for CompressedCodec<E, C> {
    fn default() -> Self {
        Self {
            inner: E::default(),
            frame: CompressFrame::default(),
        }
    }
}

impl<T, E, C> CodecEngine<T> for CompressedCodec<E, C>
where
    E: CodecEngine<T>,
    C: Compress,
{
    fn encode(&mut self, value: &T) -> std::result::Result<&[u8], SkipReason> {
        let Self { inner, frame, .. } = self;
        let raw = inner.encode(value)?;
        frame.compress(raw)
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

    /// Assert that `encoded` carries a raw (uncompressed) frame header.
    pub fn assert_raw_header(encoded: &[u8]) {
        assert!(!encoded.is_empty());
        assert_eq!(encoded[0] & 0b1111_0000, 0b0001_0000);
        assert_eq!(encoded[0] & 0b0000_1111, 0);
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::{fixtures::*, *};
    use crate::cache::codec::{CodecEngine, SkipReason};

    struct ShortDecodeCompress;

    impl Compress for ShortDecodeCompress {
        const ALGORITHM_ID: u8 = 14;

        fn max_output_size(input_len: usize) -> usize {
            input_len
        }

        fn compress_into(input: &[u8], output: &mut [u8]) -> usize {
            output[..input.len()].copy_from_slice(input);
            input.len()
        }

        fn decompress_into(_input: &[u8], output: &mut [u8]) -> Result<usize, Error> {
            if output.is_empty() {
                Ok(0)
            } else {
                Ok(output.len() - 1)
            }
        }
    }

    struct TestCompress;

    impl Compress for TestCompress {
        const ALGORITHM_ID: u8 = 15;

        fn max_output_size(input_len: usize) -> usize {
            input_len.max(1)
        }

        fn compress_into(input: &[u8], output: &mut [u8]) -> usize {
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

        fn decompress_into(input: &[u8], output: &mut [u8]) -> Result<usize, Error> {
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
                _ => Err(Error::TruncatedInput),
            }
        }
    }

    mod header {
        use super::*;

        #[test]
        fn new_compressed_roundtrip() {
            let header = Header::new_compressed(15);
            assert_eq!(header.version(), Header::VERSION_V1);
            assert_eq!(header.algorithm(), 15);
        }

        #[test]
        #[should_panic(expected = "algorithm id out of range")]
        fn new_compressed_rejects_zero() {
            let _ = Header::new_compressed(0);
        }

        #[test]
        #[should_panic(expected = "algorithm id out of range")]
        fn new_compressed_rejects_too_large() {
            let _ = Header::new_compressed(16);
        }
    }

    mod encode {
        use super::*;

        type Engine = CompressedCodec<PassthroughBytesEngine, TestCompress>;
        type OversizeEngine = CompressedCodec<SizedBytesEngine, TestCompress>;

        fn encode_decode<T>(value: &T) -> Result<(Vec<u8>, T)>
        where
            Engine: CodecEngine<T> + Default,
        {
            let mut engine = Engine::default();
            let encoded = engine.encode(value).unwrap().to_vec();
            let decoded = engine.decode(&encoded)?;
            Ok((encoded, decoded))
        }

        #[test]
        fn small_value_stays_uncompressed() -> Result<()> {
            let value = vec![7u8; 1024];
            let (encoded, decoded) = encode_decode(&value)?;
            assert_raw_header(&encoded);
            assert_eq!(decoded, value);
            Ok(())
        }

        #[test]
        fn large_compressible_value_is_compressed() -> Result<()> {
            let value = vec![b'a'; 96 * 1024];
            let (encoded, decoded) = encode_decode(&value)?;
            assert!(!encoded.is_empty());
            assert_eq!(encoded[0] & 0b1111_0000, 0b0001_0000);
            assert_eq!(encoded[0] & 0b0000_1111, TestCompress::ALGORITHM_ID);
            assert_eq!(decoded, value);
            Ok(())
        }

        #[test]
        fn incompressible_data_falls_back_to_raw() -> Result<()> {
            let value: Vec<u8> = (0..(96 * 1024)).map(|i| (i % 251) as u8).collect();
            let (encoded, decoded) = encode_decode(&value)?;
            assert_raw_header(&encoded);
            assert_eq!(decoded, value);
            Ok(())
        }

        #[test]
        fn oversize_raw_payload_skips_cache_encoding() {
            let oversize_len = CompressFrame::<TestCompress>::MAX_DECOMPRESSED_LEN + 1;
            let mut engine = OversizeEngine::default();
            assert!(matches!(
                engine.encode(&oversize_len),
                Err(SkipReason::EncodedValueTooLarge { .. })
            ));
        }
    }

    mod decode {
        use super::*;

        type Engine = CompressedCodec<PassthroughBytesEngine, TestCompress>;
        type ShortDecodeEngine = CompressedCodec<PassthroughBytesEngine, ShortDecodeCompress>;

        fn encode_bytes<T>(value: &T) -> Vec<u8>
        where
            Engine: CodecEngine<T> + Default,
        {
            let mut engine = Engine::default();
            engine.encode(value).unwrap().to_vec()
        }

        fn compressed_frame(header: Header, original_len: u32, payload: &[u8]) -> Vec<u8> {
            let mut bytes = vec![header.into_inner()];
            bytes.extend_from_slice(&original_len.to_le_bytes());
            bytes.extend_from_slice(payload);
            let checksum = crc32c::crc32c(&bytes).to_le_bytes();
            bytes.extend_from_slice(&checksum);
            bytes
        }

        fn raw_frame(payload: &[u8]) -> Vec<u8> {
            let mut bytes = vec![Header::RAW.into_inner()];
            bytes.extend_from_slice(payload);
            let checksum = crc32c::crc32c(&bytes).to_le_bytes();
            bytes.extend_from_slice(&checksum);
            bytes
        }

        fn decode_compress_error<E, T>(bytes: &[u8]) -> Error
        where
            E: CodecEngine<T> + Default,
        {
            let mut engine = E::default();
            match engine.decode(bytes) {
                Err(CodecError::Compress(err)) => err,
                Err(err) => panic!("expected compress error, got {err:?}"),
                Ok(_) => panic!("expected decode error"),
            }
        }

        #[test]
        fn corrupted_compressed_payload_returns_checksum_mismatch() {
            let value = vec![b'a'; 96 * 1024];
            let mut encoded = encode_bytes(&value);
            let payload_index = CompressFrame::<TestCompress>::COMPRESSED_PREFIX_LEN;
            encoded[payload_index] ^= 0x01;

            assert!(matches!(
                decode_compress_error::<Engine, Vec<u8>>(&encoded),
                Error::ChecksumMismatch
            ));
        }

        #[test]
        fn corrupted_declared_length_returns_checksum_mismatch() {
            let value = vec![b'a'; 96 * 1024];
            let mut encoded = encode_bytes(&value);
            encoded[1] ^= 0x01;

            assert!(matches!(
                decode_compress_error::<Engine, Vec<u8>>(&encoded),
                Error::ChecksumMismatch
            ));
        }

        #[test]
        fn compressed_header_flip_to_raw_returns_checksum_mismatch() {
            let value = vec![b'a'; 96 * 1024];
            let mut encoded = encode_bytes(&value);
            encoded[0] = Header::RAW.into_inner();

            assert!(matches!(
                decode_compress_error::<Engine, Vec<u8>>(&encoded),
                Error::ChecksumMismatch
            ));
        }

        #[test]
        fn decompressed_length_mismatch_returns_error() {
            let header = Header::new_compressed(ShortDecodeCompress::ALGORITHM_ID);
            let bytes = compressed_frame(header, 8, &[1, 2, 3, 4]);
            assert!(matches!(
                decode_compress_error::<ShortDecodeEngine, Vec<u8>>(&bytes),
                Error::DecompressedLengthMismatch
            ));
        }

        #[test]
        fn unsupported_version_returns_error() {
            assert!(matches!(
                decode_compress_error::<Engine, Vec<u8>>(&[0b0010_0000]),
                Error::UnsupportedVersion(2)
            ));
        }

        #[test]
        fn truncated_compressed_frame_returns_error() {
            let header = Header::new_compressed(TestCompress::ALGORITHM_ID);
            assert!(matches!(
                decode_compress_error::<Engine, Vec<u8>>(&[header.into_inner(), 0, 0, 0, 0,]),
                Error::TruncatedInput
            ));
        }

        #[test]
        fn algorithm_mismatch_returns_error() {
            let header = Header::new_compressed(2);
            assert!(matches!(
                decode_compress_error::<Engine, Vec<u8>>(&[header.into_inner()]),
                Error::CompressionAlgorithmMismatch
            ));
        }

        #[test]
        fn declared_length_over_limit_returns_error() {
            let header = Header::new_compressed(TestCompress::ALGORITHM_ID);
            let declared_len = CompressFrame::<TestCompress>::MAX_DECOMPRESSED_LEN as u32 + 1;
            let bytes = compressed_frame(header, declared_len, &[0xAA]);
            assert!(matches!(
                decode_compress_error::<Engine, Vec<u8>>(&bytes),
                Error::ContentTooLarge
            ));
        }

        #[test]
        fn checksum_mismatch_returns_error() {
            let header = Header::new_compressed(TestCompress::ALGORITHM_ID);
            let mut bytes = compressed_frame(header, 8, &[1, 2, 3, 4]);
            let checksum_index = bytes.len() - CompressFrame::<TestCompress>::CHECKSUM_BYTES;
            bytes[checksum_index] ^= 0x01;
            assert!(matches!(
                decode_compress_error::<Engine, Vec<u8>>(&bytes),
                Error::ChecksumMismatch
            ));
        }

        #[test]
        fn raw_checksum_mismatch_returns_error() {
            let mut bytes = raw_frame(b"raw-bytes");
            let checksum_index = bytes.len() - CompressFrame::<TestCompress>::CHECKSUM_BYTES;
            bytes[checksum_index] ^= 0x01;
            assert!(matches!(
                decode_compress_error::<Engine, Vec<u8>>(&bytes),
                Error::ChecksumMismatch
            ));
        }

        #[test]
        fn truncated_raw_frame_returns_error() {
            assert!(matches!(
                decode_compress_error::<Engine, Vec<u8>>(&[
                    Header::RAW.into_inner(),
                    0xAA,
                    0xBB,
                    0xCC,
                ]),
                Error::TruncatedInput
            ));
        }
    }
}
