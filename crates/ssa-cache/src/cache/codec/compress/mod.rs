use super::CodecEngine;
use crate::Result;

/// Compression algorithm adapter.
///
/// Implementors provide a block-level compress/decompress pair used by [`CompressedCodec`].
pub trait Compress {
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
    const COMPRESSED_HEADER_LEN: usize = 1 + Self::ORIGINAL_LEN_BYTES;
    const COMPRESSION_THRESHOLD: usize = 64 * 1024;
    const MIN_COMPRESSION_SAVINGS: usize = 4 * 1024;
    /// The length in bytes to store the original length of the uncompressed data.
    const ORIGINAL_LEN_BYTES: usize = core::mem::size_of::<u64>();

    fn encode_raw(&mut self, raw: &[u8]) -> &[u8] {
        let req_len = 1 + raw.len();
        let scratch = &mut self.scratch;
        if scratch.len() < req_len {
            scratch.resize(req_len, 0);
        }
        scratch[0] = Header::RAW.into_inner();
        scratch[1..req_len].copy_from_slice(raw);
        &scratch[..req_len]
    }

    fn compress(&mut self, raw: &[u8]) -> &[u8] {
        // If the input is not big enough to compress, return the raw data.
        if raw.len() < Self::COMPRESSION_THRESHOLD {
            return self.encode_raw(raw);
        }

        // Compress the raw data
        let max_size = C::max_output_size(raw.len());
        let compressed_frame_capacity = Self::COMPRESSED_HEADER_LEN + max_size;
        if self.scratch.len() < compressed_frame_capacity {
            self.scratch.resize(compressed_frame_capacity, 0);
        }
        let compressed_len =
            C::compress_into(raw, &mut self.scratch[Self::COMPRESSED_HEADER_LEN..]);
        let total_compressed_len = Self::COMPRESSED_HEADER_LEN + compressed_len;

        // If compression doesn't save enough bytes, return the raw data instead.
        let total_raw_len = 1 + raw.len();
        let saved_bytes = total_raw_len.saturating_sub(total_compressed_len);
        if saved_bytes < Self::MIN_COMPRESSION_SAVINGS {
            return self.encode_raw(raw);
        }

        // Write the header into the scratch buffer.
        self.scratch[0] = const { Header::new_compressed(C::ALGORITHM_ID).into_inner() };
        self.scratch[1..Self::COMPRESSED_HEADER_LEN]
            .copy_from_slice(&(raw.len() as u64).to_le_bytes());

        // Return the compressed data including the header.
        &self.scratch[..total_compressed_len]
    }

    fn decompress<'a>(&'a mut self, bytes: &'a [u8]) -> Result<&'a [u8], Error> {
        let (header_byte, payload) = bytes.split_first().ok_or(Error::HeaderMismatch)?;
        let header = Header::from_inner(*header_byte);

        let version = header.version();
        if version != Header::VERSION_V1 {
            return Err(Error::UnsupportedVersion(version));
        }

        let scratch = &mut self.scratch;
        match header.algorithm() {
            Header::ALGORITHM_RAW => Ok(payload),
            id if id == C::ALGORITHM_ID => {
                if payload.len() < Self::ORIGINAL_LEN_BYTES {
                    return Err(Error::TruncatedInput);
                }
                let (len_bytes, compressed_payload) = payload.split_at(Self::ORIGINAL_LEN_BYTES);
                let original_len = u64::from_le_bytes(
                    len_bytes
                        .try_into()
                        .expect("payload is checked to be at least ORIGINAL_LEN_BYTES"),
                );
                #[cfg(any(target_pointer_width = "16", target_pointer_width = "32"))]
                if original_len > usize::MAX as u64 {
                    return Err(Error::ContentTooLarge);
                }
                let original_len = original_len as usize;
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
/// # Bit Layout
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
    fn encode(&mut self, value: &T) -> &[u8] {
        let Self { inner, frame, .. } = self;
        let raw = inner.encode(value);
        frame.compress(raw)
    }

    fn decode(&mut self, bytes: &[u8]) -> Result<T> {
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
    pub struct BytesRaw {
        buffer: Vec<u8>,
    }

    impl CodecEngine<Vec<u8>> for BytesRaw {
        fn encode(&mut self, value: &Vec<u8>) -> &[u8] {
            self.buffer.clear();
            self.buffer.extend_from_slice(value);
            &self.buffer
        }

        fn decode(&mut self, bytes: &[u8]) -> Result<Vec<u8>> {
            Ok(bytes.to_vec())
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
    use super::{
        Compress, CompressedCodec, Error, Header,
        fixtures::{BytesRaw, assert_raw_header},
    };
    use crate::cache::codec::CodecEngine;

    #[test]
    fn test_raw_header() {
        assert_raw_header(&[Header::RAW.into_inner()]);
    }

    #[test]
    fn test_header_new_compressed_roundtrip() {
        let header = Header::new_compressed(3);
        assert_eq!(header.version(), Header::VERSION_V1);
        assert_eq!(header.algorithm(), 3);
    }

    #[test]
    #[should_panic(expected = "algorithm id out of range")]
    fn test_header_new_compressed_rejects_zero() {
        let _ = Header::new_compressed(0);
    }

    #[test]
    #[should_panic(expected = "algorithm id out of range")]
    fn test_header_new_compressed_rejects_too_large() {
        let _ = Header::new_compressed(16);
    }

    struct LenMismatchCompress;

    impl Compress for LenMismatchCompress {
        const ALGORITHM_ID: u8 = 2;

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

    #[test]
    fn decode_decompressed_length_mismatch_returns_error() {
        type Engine = CompressedCodec<BytesRaw, LenMismatchCompress>;

        let mut bytes =
            vec![Header::new_compressed(LenMismatchCompress::ALGORITHM_ID).into_inner()];
        bytes.extend_from_slice(&(8u64).to_le_bytes());
        bytes.extend_from_slice(&[1, 2, 3, 4]);

        let mut engine = Engine::default();
        let err = engine.decode(&bytes).unwrap_err();
        assert!(matches!(
            err,
            crate::Error::Compress(Error::DecompressedLengthMismatch)
        ));
    }
}
