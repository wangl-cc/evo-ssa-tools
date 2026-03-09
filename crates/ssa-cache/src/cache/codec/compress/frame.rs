//! Framed wire format and scratch-buffer management for compressed codec payloads.
//!
//! # Header Bit Layout
//!
//! ```text
//!   bit7..4  bit3..0
//! +---------+--------+
//! | version |  algo  |
//! +---------+--------+
//! ```
//!
//! - `version`: format version (`0b0001` currently).
//! - `algo`: algorithm id (`0` for uncompressed payloads). Non-zero values represent compressed
//!   payloads.
//!
//! # Wire Format
//!
//! Raw frame:
//!
//! ```text
//! [header][raw_payload...][checksum_crc32c_u32_le]
//! ```
//!
//! Raw-frame checksum coverage:
//!
//! ```text
//! crc32c(header || raw_payload)
//! ```
//!
//! Compressed frame:
//!
//! ```text
//! [header][original_len_u32_le][compressed_payload...][checksum_crc32c_u32_le]
//! ```
//!
//! Compressed-frame checksum coverage:
//!
//! ```text
//! crc32c(header || original_len_u32_le || compressed_payload)
//! ```

use core::num::NonZeroUsize;

use crc32c::crc32c;

use super::algorithm::Compress;

/// Bytes used by the trailing CRC32C checksum in both raw and compressed frames.
pub const CHECKSUM_BYTES: usize = core::mem::size_of::<u32>();
/// Bytes used by the stored original payload length in compressed frames.
pub const ORIGINAL_LEN_BYTES: usize = core::mem::size_of::<u32>();
/// Fixed framing overhead for a raw frame: header + trailing checksum.
pub const RAW_FRAME_EXTRA_LEN: usize = Header::HEADER_BYTES + CHECKSUM_BYTES;
/// Fixed framing overhead for a compressed frame: header + original length + trailing checksum.
pub const COMPRESSED_FRAME_EXTRA_LEN: usize =
    Header::HEADER_BYTES + ORIGINAL_LEN_BYTES + CHECKSUM_BYTES;

/// Decode-time failures specific to the framed compression layer.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Empty input")]
    EmptyInput,

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Header {
    value: u8,
}

impl Header {
    const HEADER_BYTES: usize = 1;
}

impl Header {
    const ALGORITHM_MASK: u8 = 0b0000_1111;
    const VERSION_MASK: u8 = 0b1111_0000;
    const VERSION_SHIFT: u8 = 4;
}

impl Header {
    /// The number of bytes used for the version field.
    const VERSION_V1: u8 = 0b001;
    /// Encoded version for version 1.
    const VERSION_V1_ENCODED: u8 = Self::VERSION_V1 << Self::VERSION_SHIFT;
}

impl Header {
    /// The reserved algorithm ID for an uncompressed payload.
    const ALGORITHM_RAW: u8 = 0;
    /// The raw header value for an uncompressed payload.
    pub(super) const RAW: Self = Self::from_inner(Self::VERSION_V1_ENCODED | Self::ALGORITHM_RAW);

    /// Creates a new compressed header with the given algorithm ID.
    pub(super) const fn new_compressed(algorithm_id: u8) -> Self {
        assert!(
            0 < algorithm_id && algorithm_id <= Self::ALGORITHM_MASK,
            "algorithm id out of range (must be 1..=15)",
        );
        let value = Self::VERSION_V1_ENCODED | (algorithm_id & Self::ALGORITHM_MASK);
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

/// Reusable framing buffer used to build and decode framed payloads.
pub(super) struct CompressFrame<C> {
    scratch: Vec<u8>,
    compressor: C,
    max_decode_len: Option<NonZeroUsize>,
}

impl<C> CompressFrame<C> {
    pub(super) fn new(compressor: C) -> Self {
        Self {
            scratch: Vec::new(),
            compressor,
            max_decode_len: None,
        }
    }

    pub(super) fn with_compressor(mut self, compressor: C) -> Self {
        self.compressor = compressor;
        self
    }

    pub(super) fn with_max_decode_len(mut self, max_decode_len: Option<NonZeroUsize>) -> Self {
        self.max_decode_len = max_decode_len;
        self
    }
}

impl<C> CompressFrame<C> {
    /// The length of the prefix of compressed frame (`[header][orig_len_u32_le]`)
    const COMPRESSED_PREFIX_LEN: usize = Header::HEADER_BYTES + ORIGINAL_LEN_BYTES;
    /// The length of the prefix of raw (uncompressed) frames (`[header]`).
    const RAW_PREFIX_LEN: usize = Header::HEADER_BYTES;
}

impl<C: Compress> CompressFrame<C> {
    /// Encode a raw (uncompressed) frame into the scratch buffer.
    ///
    /// The frame size of the encoded frame is `raw.len() + RAW_FRAME_EXTRA_LEN`.
    /// It can be accessed using [`frame`](Self::frame).
    pub(super) fn encode_raw(&mut self, raw: &[u8]) {
        let total_len = raw.len() + RAW_FRAME_EXTRA_LEN;
        let scratch = &mut self.scratch;
        if scratch.len() < total_len {
            scratch.resize(total_len, 0);
        }
        scratch[0] = const { Header::RAW.into_inner() };
        scratch[Self::RAW_PREFIX_LEN..Self::RAW_PREFIX_LEN + raw.len()].copy_from_slice(raw);
        let checksum = crc32c(&scratch[..Self::RAW_PREFIX_LEN + raw.len()]);
        scratch[Self::RAW_PREFIX_LEN + raw.len()..total_len]
            .copy_from_slice(&checksum.to_le_bytes());
    }

    /// Encode a compressed frame into the scratch buffer.
    ///
    /// The frame size of the encoded frame is `compressed_len + COMPRESSED_FRAME_EXTRA_LEN`.
    /// It can be accessed using [`frame`](Self::frame).
    ///
    /// # Panics
    ///
    /// Payloads larger than `u32::MAX` are outside the supported framed format
    /// and will panic here when writing the stored original length field.
    pub(super) fn encode_compressed(&mut self, raw: &[u8]) -> usize {
        let original_len = u32::try_from(raw.len())
            .expect("compressed frame header requires raw.len() <= u32::MAX");

        let max_size = self.compressor.max_output_size(raw.len());
        let compressed_frame_capacity = Self::COMPRESSED_PREFIX_LEN + max_size + CHECKSUM_BYTES;
        if self.scratch.len() < compressed_frame_capacity {
            self.scratch.resize(compressed_frame_capacity, 0);
        }
        let compressed_len = self.compressor.compress_into(
            raw,
            &mut self.scratch[Self::COMPRESSED_PREFIX_LEN..Self::COMPRESSED_PREFIX_LEN + max_size],
        );
        let total_compressed_len = compressed_len + COMPRESSED_FRAME_EXTRA_LEN;

        self.scratch[0] = const { Header::new_compressed(C::ALGORITHM_ID).into_inner() };
        let original_len_bytes = original_len.to_le_bytes();
        self.scratch[1..Self::COMPRESSED_PREFIX_LEN].copy_from_slice(&original_len_bytes);
        let checksum = crc32c(&self.scratch[..Self::COMPRESSED_PREFIX_LEN + compressed_len]);
        self.scratch[Self::COMPRESSED_PREFIX_LEN + compressed_len..total_compressed_len]
            .copy_from_slice(&checksum.to_le_bytes());

        total_compressed_len
    }

    /// Get the frame bytes for the given size.
    pub(super) fn get_frame(&self, frame_size: usize) -> &[u8] {
        &self.scratch[..frame_size]
    }

    /// Decode a framed payload into raw serialized bytes for the inner codec.
    ///
    /// Raw frames borrow directly from the input slice. Compressed frames decompress into the
    /// reusable scratch buffer owned by this frame.
    pub(super) fn decompress<'a>(&'a mut self, bytes: &'a [u8]) -> Result<&'a [u8], Error> {
        let (header_byte, payload) = bytes.split_first().ok_or(Error::EmptyInput)?;
        let header = Header::from_inner(*header_byte);

        let version = header.version();
        if version != Header::VERSION_V1 {
            return Err(Error::UnsupportedVersion(version));
        }

        let scratch = &mut self.scratch;
        match header.algorithm() {
            Header::ALGORITHM_RAW => {
                if payload.len() < CHECKSUM_BYTES {
                    return Err(Error::TruncatedInput);
                }
                let (raw_payload, checksum_bytes) =
                    payload.split_at(payload.len() - CHECKSUM_BYTES);
                let stored_checksum = u32::from_le_bytes(
                    checksum_bytes
                        .try_into()
                        .expect("payload is checked to be at least CHECKSUM_BYTES"),
                );
                let actual_checksum = crc32c(&bytes[..bytes.len() - CHECKSUM_BYTES]);
                if actual_checksum != stored_checksum {
                    return Err(Error::ChecksumMismatch);
                }
                Ok(raw_payload)
            }
            id if id == C::ALGORITHM_ID => {
                if payload.len() < ORIGINAL_LEN_BYTES + CHECKSUM_BYTES {
                    return Err(Error::TruncatedInput);
                }
                let (len_bytes, rest) = payload.split_at(ORIGINAL_LEN_BYTES);
                let (compressed_payload, checksum_bytes) =
                    rest.split_at(rest.len() - CHECKSUM_BYTES);
                let stored_checksum = u32::from_le_bytes(
                    checksum_bytes
                        .try_into()
                        .expect("payload is checked to be at least CHECKSUM_BYTES"),
                );
                let actual_checksum = crc32c(&bytes[..bytes.len() - CHECKSUM_BYTES]);
                if actual_checksum != stored_checksum {
                    return Err(Error::ChecksumMismatch);
                }
                let original_len = u32::from_le_bytes(
                    len_bytes
                        .try_into()
                        .expect("payload is checked to be at least ORIGINAL_LEN_BYTES"),
                ) as usize;
                if let Some(max_decode_len) = self.max_decode_len
                    && original_len > max_decode_len.get()
                {
                    return Err(Error::ContentTooLarge);
                }
                if scratch.len() < original_len {
                    scratch.resize(original_len, 0);
                }
                let decoded_len = self
                    .compressor
                    .decompress_into(compressed_payload, &mut scratch[..original_len])?;
                if decoded_len != original_len {
                    return Err(Error::DecompressedLengthMismatch);
                }
                Ok(&scratch[..decoded_len])
            }
            _ => Err(Error::CompressionAlgorithmMismatch),
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
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

    #[derive(Default)]
    struct ShortDecodeCompress;

    impl Compress for ShortDecodeCompress {
        const ALGORITHM_ID: u8 = 14;

        fn max_output_size(&self, input_len: usize) -> usize {
            input_len
        }

        fn compress_into(&self, input: &[u8], output: &mut [u8]) -> usize {
            output[..input.len()].copy_from_slice(input);
            input.len()
        }

        fn decompress_into(&self, _input: &[u8], output: &mut [u8]) -> Result<usize, Error> {
            if output.is_empty() {
                Ok(0)
            } else {
                Ok(output.len() - 1)
            }
        }
    }

    #[derive(Default)]
    struct TestCompress;

    impl Compress for TestCompress {
        const ALGORITHM_ID: u8 = 15;

        fn max_output_size(&self, input_len: usize) -> usize {
            input_len.max(1)
        }

        fn compress_into(&self, input: &[u8], output: &mut [u8]) -> usize {
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

        fn decompress_into(&self, input: &[u8], output: &mut [u8]) -> Result<usize, Error> {
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

    fn compressed_frame(header: Header, original_len: u32, payload: &[u8]) -> Vec<u8> {
        let mut bytes = vec![header.into_inner()];
        bytes.extend_from_slice(&original_len.to_le_bytes());
        bytes.extend_from_slice(payload);
        let checksum = crc32c(&bytes).to_le_bytes();
        bytes.extend_from_slice(&checksum);
        bytes
    }

    fn raw_frame(payload: &[u8]) -> Vec<u8> {
        let mut bytes = vec![Header::RAW.into_inner()];
        bytes.extend_from_slice(payload);
        let checksum = crc32c(&bytes).to_le_bytes();
        bytes.extend_from_slice(&checksum);
        bytes
    }

    fn decode_frame_error<C>(
        bytes: &[u8],
        compressor: C,
        max_decode_len: Option<NonZeroUsize>,
    ) -> Error
    where
        C: Compress,
    {
        let mut frame = CompressFrame::new(compressor).with_max_decode_len(max_decode_len);
        match frame.decompress(bytes) {
            Err(err) => err,
            Ok(_) => panic!("expected decode error"),
        }
    }

    #[test]
    fn compressed_frame_checksum_coverage_detects_corruption() {
        let header = Header::new_compressed(TestCompress::ALGORITHM_ID);
        let base = compressed_frame(header, 96 * 1024, b"a");
        let payload_index = CompressFrame::<TestCompress>::COMPRESSED_PREFIX_LEN;

        for encoded in [
            {
                let mut encoded = base.clone();
                encoded[payload_index] ^= 0x01;
                encoded
            },
            {
                let mut encoded = base.clone();
                encoded[1] ^= 0x01;
                encoded
            },
            {
                let mut encoded = base.clone();
                encoded[0] = Header::RAW.into_inner();
                encoded
            },
        ] {
            assert!(matches!(
                decode_frame_error(&encoded, TestCompress, None),
                Error::ChecksumMismatch
            ));
        }
    }

    #[test]
    fn decompressed_length_mismatch_returns_error() {
        let header = Header::new_compressed(ShortDecodeCompress::ALGORITHM_ID);
        let bytes = compressed_frame(header, 8, &[1, 2, 3, 4]);
        assert!(matches!(
            decode_frame_error(&bytes, ShortDecodeCompress, None),
            Error::DecompressedLengthMismatch
        ));
    }

    #[test]
    fn unsupported_version_returns_error() {
        assert!(matches!(
            decode_frame_error(&[0b0010_0000], TestCompress, None),
            Error::UnsupportedVersion(2)
        ));
    }

    #[test]
    fn empty_input_returns_error() {
        assert!(matches!(
            decode_frame_error(&[], TestCompress, None),
            Error::EmptyInput
        ));
    }

    #[test]
    fn truncated_compressed_frame_returns_error() {
        let header = Header::new_compressed(TestCompress::ALGORITHM_ID);
        assert!(matches!(
            decode_frame_error(&[header.into_inner(), 0, 0, 0, 0], TestCompress, None),
            Error::TruncatedInput
        ));
    }

    #[test]
    fn algorithm_mismatch_returns_error() {
        let header = Header::new_compressed(2);
        assert!(matches!(
            decode_frame_error(&[header.into_inner()], TestCompress, None),
            Error::CompressionAlgorithmMismatch
        ));
    }

    #[test]
    fn declared_length_over_decode_limit_returns_error() {
        let header = Header::new_compressed(TestCompress::ALGORITHM_ID);
        let declared_len = 9;
        let bytes = compressed_frame(header, declared_len, &[0xAA]);
        assert!(matches!(
            decode_frame_error(&bytes, TestCompress, NonZeroUsize::new(8)),
            Error::ContentTooLarge
        ));
    }

    #[test]
    fn checksum_mismatch_returns_error() {
        let header = Header::new_compressed(TestCompress::ALGORITHM_ID);
        let mut bytes = compressed_frame(header, 8, &[1, 2, 3, 4]);
        let checksum_index = bytes.len() - CHECKSUM_BYTES;
        bytes[checksum_index] ^= 0x01;
        assert!(matches!(
            decode_frame_error(&bytes, TestCompress, None),
            Error::ChecksumMismatch
        ));
    }

    #[test]
    fn raw_checksum_mismatch_returns_error() {
        let mut bytes = raw_frame(b"raw-bytes");
        let checksum_index = bytes.len() - CHECKSUM_BYTES;
        bytes[checksum_index] ^= 0x01;
        assert!(matches!(
            decode_frame_error(&bytes, TestCompress, None),
            Error::ChecksumMismatch
        ));
    }

    #[test]
    fn truncated_raw_frame_returns_error() {
        assert!(matches!(
            decode_frame_error(
                &[Header::RAW.into_inner(), 0xAA, 0xBB, 0xCC],
                TestCompress,
                None
            ),
            Error::TruncatedInput
        ));
    }

    #[test]
    fn raw_frame_ignores_decode_limit() -> Result<(), Error> {
        let bytes = raw_frame(b"raw-bytes");
        let mut frame = CompressFrame::new(TestCompress).with_max_decode_len(NonZeroUsize::new(1));
        let decoded = frame.decompress(&bytes)?;
        assert_eq!(decoded, b"raw-bytes");
        Ok(())
    }
}
