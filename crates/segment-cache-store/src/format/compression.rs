//! Value-payload compression formats and block-local payload frames.
//!
//! Compression is a physical block-level concern. Keys and value indexes stay
//! uncompressed so ordered lookup can validate and search metadata before
//! deciding whether the value payload needs to be read.

use crate::format::{CorruptionError, FormatError, format_u32};

/// Default raw payload length before the writer attempts value-payload compression.
pub const DEFAULT_VALUE_PAYLOAD_COMPRESSION_MIN_TRY_LEN: usize = 64 * 1024;
/// Default minimum saved percentage before a compressed frame is kept.
pub const DEFAULT_VALUE_PAYLOAD_COMPRESSION_MIN_SAVED_PERCENT: u8 = 20;

#[cfg(any(feature = "value-compression-lz4", feature = "value-compression-zstd"))]
pub(crate) const VALUE_PAYLOAD_FRAME_HEADER_LEN: usize = 12;

#[cfg(any(feature = "value-compression-lz4", feature = "value-compression-zstd"))]
const VALUE_PAYLOAD_ENCODING_RAW: u32 = 0;
#[cfg(feature = "value-compression-lz4")]
const VALUE_PAYLOAD_ENCODING_LZ4: u32 = 1;
#[cfg(feature = "value-compression-zstd")]
const VALUE_PAYLOAD_ENCODING_ZSTD: u32 = 2;

/// Store-wide value-payload compression policy.
///
/// The policy is persisted in `STORE` and in every segment header. When a
/// compression-capable policy is enabled, each block still records whether its
/// payload was stored raw or compressed; payloads that are too small or do not
/// compress well stay raw.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValuePayloadCompressionKind {
    /// Store value payload bytes exactly as provided.
    None,

    /// LZ4 block compression for value payloads.
    #[cfg(feature = "value-compression-lz4")]
    Lz4,

    /// Zstandard level-1 compression for value payloads.
    #[cfg(feature = "value-compression-zstd")]
    ZstdLevel1,
}

/// Reusable value-payload compression context for segment writes.
///
/// LZ4 and raw frames are stateless. Zstandard keeps a reusable compression
/// context so one segment publish does not recreate it for every block.
pub(crate) struct ValuePayloadEncoder {
    #[cfg(any(feature = "value-compression-lz4", feature = "value-compression-zstd"))]
    kind: ValuePayloadCompressionKind,
    #[cfg(feature = "value-compression-zstd")]
    zstd: Option<zstd::bulk::Compressor<'static>>,
    #[cfg(feature = "value-compression-zstd")]
    scratch: Vec<u8>,
}

/// Reusable value-payload decompression context for ordered reads and scans.
///
/// The decoder owns algorithm context and one reusable decoded-payload buffer.
/// Decoded blocks borrow raw payload bytes when possible; when a compressed
/// frame must be materialized, the buffer is moved into the block and reclaimed
/// when the block is evicted from the cursor.
pub(crate) struct ValuePayloadDecoder {
    #[cfg(feature = "value-compression-zstd")]
    zstd: Option<zstd::bulk::Decompressor<'static>>,
    #[cfg(any(feature = "value-compression-lz4", feature = "value-compression-zstd"))]
    buffer: Vec<u8>,
}

/// Writer-side policy for deciding whether a value payload is worth compressing.
///
/// This policy is not persisted. It only controls newly written blocks. The
/// persisted store compression kind still determines which frame encodings a
/// reader must support, and each block records whether its own payload was
/// stored raw or compressed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ValuePayloadCompressionPolicy {
    /// Minimum raw value-payload bytes before compression is attempted.
    pub min_try_len: usize,
    /// Minimum saved percentage required to keep a compressed frame.
    pub min_saved_percent: u8,
}

impl Default for ValuePayloadCompressionPolicy {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl ValuePayloadCompressionPolicy {
    /// Default writer-side compression policy used by [`crate::CommitOptions`].
    pub const DEFAULT: Self = Self {
        min_try_len: DEFAULT_VALUE_PAYLOAD_COMPRESSION_MIN_TRY_LEN,
        min_saved_percent: DEFAULT_VALUE_PAYLOAD_COMPRESSION_MIN_SAVED_PERCENT,
    };

    /// Creates a policy from explicit thresholds.
    pub const fn new(min_try_len: usize, min_saved_percent: u8) -> Self {
        Self {
            min_try_len,
            min_saved_percent,
        }
    }

    /// Sets the minimum raw value-payload bytes before compression is attempted.
    pub fn with_min_try_len(mut self, min_try_len: usize) -> Self {
        self.min_try_len = min_try_len;
        self
    }

    /// Sets the minimum saved percentage required to keep a compressed frame.
    pub fn with_min_saved_percent(mut self, min_saved_percent: u8) -> Self {
        self.min_saved_percent = min_saved_percent;
        self
    }

    pub(crate) fn min_saved_percent_is_valid(self) -> bool {
        self.min_saved_percent <= 100
    }

    #[cfg(any(feature = "value-compression-lz4", feature = "value-compression-zstd"))]
    fn should_try(self, raw_len: usize) -> bool {
        raw_len >= self.min_try_len
    }

    #[cfg(any(feature = "value-compression-lz4", feature = "value-compression-zstd"))]
    fn should_keep_compressed(self, raw_len: usize, compressed_len: usize) -> bool {
        if compressed_len >= raw_len {
            return false;
        }
        let saved = raw_len - compressed_len;
        saved.saturating_mul(100) >= raw_len.saturating_mul(usize::from(self.min_saved_percent))
    }
}

impl ValuePayloadCompressionKind {
    /// Default value-payload compression used by [`crate::CreateOptions::new`].
    pub const DEFAULT: Self = Self::None;

    /// Resolves a persisted compression id to a built-in policy.
    pub const fn from_format_id(format_id: u32) -> Option<Self> {
        match format_id {
            0 => Some(Self::None),
            #[cfg(feature = "value-compression-lz4")]
            1 => Some(Self::Lz4),
            #[cfg(feature = "value-compression-zstd")]
            2 => Some(Self::ZstdLevel1),
            _ => None,
        }
    }

    /// Human-readable compression name used in diagnostics and debug output.
    pub const fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            #[cfg(feature = "value-compression-lz4")]
            Self::Lz4 => "lz4",
            #[cfg(feature = "value-compression-zstd")]
            Self::ZstdLevel1 => "zstd_level1",
        }
    }

    /// Stable persisted compression identifier.
    pub const fn format_id(self) -> u32 {
        match self {
            Self::None => 0,
            #[cfg(feature = "value-compression-lz4")]
            Self::Lz4 => 1,
            #[cfg(feature = "value-compression-zstd")]
            Self::ZstdLevel1 => 2,
        }
    }

    /// Bytes stored before the encoded payload for this policy.
    pub const fn frame_header_len(self) -> usize {
        match self {
            Self::None => 0,
            #[cfg(feature = "value-compression-lz4")]
            Self::Lz4 => VALUE_PAYLOAD_FRAME_HEADER_LEN,
            #[cfg(feature = "value-compression-zstd")]
            Self::ZstdLevel1 => VALUE_PAYLOAD_FRAME_HEADER_LEN,
        }
    }

    pub(crate) fn parse_frame_header(
        self,
        bytes: &[u8],
        expected_raw_len: usize,
    ) -> Result<ValuePayloadFrame, CorruptionError> {
        #[cfg(not(any(feature = "value-compression-lz4", feature = "value-compression-zstd")))]
        let _ = bytes;

        match self {
            Self::None => Ok(ValuePayloadFrame::raw_without_header(expected_raw_len)
                .map_err(|_| CorruptionError::Block)?),
            #[cfg(feature = "value-compression-lz4")]
            Self::Lz4 => ValuePayloadFrame::from_bytes(bytes, expected_raw_len),
            #[cfg(feature = "value-compression-zstd")]
            Self::ZstdLevel1 => ValuePayloadFrame::from_bytes(bytes, expected_raw_len),
        }
    }

    pub(crate) fn decode_frame<'frame>(
        self,
        decoder: &mut ValuePayloadDecoder,
        frame: &'frame [u8],
        expected_raw_len: usize,
    ) -> Result<DecodedPayload<'frame>, CorruptionError> {
        decoder.decode_frame(self, frame, expected_raw_len)
    }
}

impl ValuePayloadEncoder {
    pub(crate) fn new(kind: ValuePayloadCompressionKind) -> Self {
        #[cfg(not(any(feature = "value-compression-lz4", feature = "value-compression-zstd")))]
        let _ = kind;

        Self {
            #[cfg(any(feature = "value-compression-lz4", feature = "value-compression-zstd"))]
            kind,
            #[cfg(feature = "value-compression-zstd")]
            zstd: match kind {
                ValuePayloadCompressionKind::ZstdLevel1 => Some(
                    zstd::bulk::Compressor::new(1)
                        .expect("zstd level 1 compressor context should be constructible"),
                ),
                _ => None,
            },
            #[cfg(feature = "value-compression-zstd")]
            scratch: Vec::new(),
        }
    }

    #[cfg(any(feature = "value-compression-lz4", feature = "value-compression-zstd"))]
    pub(crate) fn encode_frame(
        &mut self,
        raw_payload: &[u8],
        policy: ValuePayloadCompressionPolicy,
        out: &mut Vec<u8>,
    ) -> Result<ValuePayloadFrame, FormatError> {
        #[cfg(not(any(feature = "value-compression-lz4", feature = "value-compression-zstd")))]
        let _ = policy;

        match self.kind {
            ValuePayloadCompressionKind::None => {
                out.extend_from_slice(raw_payload);
                Ok(ValuePayloadFrame::raw_without_header(raw_payload.len())?)
            }
            #[cfg(feature = "value-compression-lz4")]
            ValuePayloadCompressionKind::Lz4 => encode_lz4_frame(raw_payload, policy, out),
            #[cfg(feature = "value-compression-zstd")]
            ValuePayloadCompressionKind::ZstdLevel1 => encode_zstd_frame(
                raw_payload,
                policy,
                out,
                self.zstd.as_mut(),
                &mut self.scratch,
            ),
        }
    }
}

impl ValuePayloadDecoder {
    pub(crate) fn new(kind: ValuePayloadCompressionKind) -> Self {
        #[cfg(not(feature = "value-compression-zstd"))]
        let _ = kind;

        Self {
            #[cfg(feature = "value-compression-zstd")]
            zstd: match kind {
                ValuePayloadCompressionKind::ZstdLevel1 => Some(
                    zstd::bulk::Decompressor::new()
                        .expect("zstd decompressor context should be constructible"),
                ),
                _ => None,
            },
            #[cfg(any(feature = "value-compression-lz4", feature = "value-compression-zstd"))]
            buffer: Vec::new(),
        }
    }

    pub(crate) fn reclaim_payload_buffer(&mut self, buffer: Option<Vec<u8>>) {
        #[cfg(not(any(feature = "value-compression-lz4", feature = "value-compression-zstd")))]
        let _ = buffer;

        #[cfg(any(feature = "value-compression-lz4", feature = "value-compression-zstd"))]
        {
            let Some(buffer) = buffer else {
                return;
            };
            if buffer.capacity() > self.buffer.capacity() {
                self.buffer = buffer;
            }
        }
    }

    pub(crate) fn decode_frame<'frame>(
        &mut self,
        kind: ValuePayloadCompressionKind,
        frame: &'frame [u8],
        expected_raw_len: usize,
    ) -> Result<DecodedPayload<'frame>, CorruptionError> {
        let header = kind.parse_frame_header(frame, expected_raw_len)?;
        let encoded = frame
            .get(header.encoded_range())
            .ok_or(CorruptionError::Block)?;
        match header.encoding {
            ValuePayloadEncoding::Raw => Ok(DecodedPayload::Borrowed(encoded)),
            #[cfg(feature = "value-compression-lz4")]
            ValuePayloadEncoding::Lz4 => {
                self.buffer.resize(expected_raw_len, 0);
                let written = lz4_flex::block::decompress_into(encoded, &mut self.buffer)
                    .map_err(|_| CorruptionError::Block)?;
                if written != expected_raw_len {
                    return Err(CorruptionError::Block);
                }
                let decoded = std::mem::take(&mut self.buffer);
                Ok(DecodedPayload::Owned(decoded))
            }
            #[cfg(feature = "value-compression-zstd")]
            ValuePayloadEncoding::Zstd => {
                self.buffer.resize(expected_raw_len, 0);
                let written = self
                    .zstd
                    .as_mut()
                    .ok_or(CorruptionError::Block)?
                    .decompress_to_buffer(encoded, &mut self.buffer)
                    .map_err(|_| CorruptionError::Block)?;
                if written != expected_raw_len {
                    return Err(CorruptionError::Block);
                }
                let decoded = std::mem::take(&mut self.buffer);
                Ok(DecodedPayload::Owned(decoded))
            }
        }
    }
}

pub(crate) enum DecodedPayload<'a> {
    Borrowed(&'a [u8]),
    #[cfg(any(feature = "value-compression-lz4", feature = "value-compression-zstd"))]
    Owned(Vec<u8>),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ValuePayloadFrame {
    encoding: ValuePayloadEncoding,
    raw_len: usize,
    encoded_len: usize,
    header_len: usize,
}

impl ValuePayloadFrame {
    fn raw_without_header(raw_len: usize) -> Result<Self, FormatError> {
        let raw_len = format_u32(raw_len, "block value payload length")? as usize;
        Ok(Self {
            encoding: ValuePayloadEncoding::Raw,
            raw_len,
            encoded_len: raw_len,
            header_len: 0,
        })
    }

    #[cfg(any(feature = "value-compression-lz4", feature = "value-compression-zstd"))]
    fn raw_with_header(raw_len: usize) -> Result<Self, FormatError> {
        let raw_len = format_u32(raw_len, "block value payload length")? as usize;
        Ok(Self {
            encoding: ValuePayloadEncoding::Raw,
            raw_len,
            encoded_len: raw_len,
            header_len: VALUE_PAYLOAD_FRAME_HEADER_LEN,
        })
    }

    #[cfg(feature = "value-compression-lz4")]
    fn lz4(raw_len: usize, encoded_len: usize) -> Result<Self, FormatError> {
        let raw_len = format_u32(raw_len, "block value payload length")? as usize;
        let encoded_len =
            format_u32(encoded_len, "compressed block value payload length")? as usize;
        Ok(Self {
            encoding: ValuePayloadEncoding::Lz4,
            raw_len,
            encoded_len,
            header_len: VALUE_PAYLOAD_FRAME_HEADER_LEN,
        })
    }

    #[cfg(feature = "value-compression-zstd")]
    fn zstd(raw_len: usize, encoded_len: usize) -> Result<Self, FormatError> {
        let raw_len = format_u32(raw_len, "block value payload length")? as usize;
        let encoded_len =
            format_u32(encoded_len, "compressed block value payload length")? as usize;
        Ok(Self {
            encoding: ValuePayloadEncoding::Zstd,
            raw_len,
            encoded_len,
            header_len: VALUE_PAYLOAD_FRAME_HEADER_LEN,
        })
    }

    #[cfg(any(feature = "value-compression-lz4", feature = "value-compression-zstd"))]
    fn from_bytes(bytes: &[u8], expected_raw_len: usize) -> Result<Self, CorruptionError> {
        if bytes.len() < VALUE_PAYLOAD_FRAME_HEADER_LEN {
            return Err(CorruptionError::Block);
        }
        let encoding_id = read_u32(bytes, 0)?;
        let raw_len = read_u32(bytes, 4)? as usize;
        let encoded_len = read_u32(bytes, 8)? as usize;
        if raw_len != expected_raw_len {
            return Err(CorruptionError::Block);
        }
        let encoding = match encoding_id {
            VALUE_PAYLOAD_ENCODING_RAW => ValuePayloadEncoding::Raw,
            #[cfg(feature = "value-compression-lz4")]
            VALUE_PAYLOAD_ENCODING_LZ4 => ValuePayloadEncoding::Lz4,
            #[cfg(feature = "value-compression-zstd")]
            VALUE_PAYLOAD_ENCODING_ZSTD => ValuePayloadEncoding::Zstd,
            _ => return Err(CorruptionError::Block),
        };
        if encoding == ValuePayloadEncoding::Raw && encoded_len != raw_len {
            return Err(CorruptionError::Block);
        }
        Ok(Self {
            encoding,
            raw_len,
            encoded_len,
            header_len: VALUE_PAYLOAD_FRAME_HEADER_LEN,
        })
    }

    #[cfg(any(feature = "value-compression-lz4", feature = "value-compression-zstd"))]
    pub(crate) fn write_header(self, out: &mut Vec<u8>) {
        if self.header_len == 0 {
            return;
        }
        out.extend_from_slice(&self.encoding.format_id().to_le_bytes());
        out.extend_from_slice(
            &u32::try_from(self.raw_len)
                .expect("raw payload length was formatted")
                .to_le_bytes(),
        );
        out.extend_from_slice(
            &u32::try_from(self.encoded_len)
                .expect("encoded payload length was formatted")
                .to_le_bytes(),
        );
    }

    pub(crate) fn frame_len(self) -> usize {
        self.header_len + self.encoded_len
    }

    pub(crate) fn encoded_range(self) -> std::ops::Range<usize> {
        self.header_len..self.header_len + self.encoded_len
    }

    pub(crate) fn is_raw_borrowable(self) -> bool {
        self.encoding == ValuePayloadEncoding::Raw
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ValuePayloadEncoding {
    Raw,
    #[cfg(feature = "value-compression-lz4")]
    Lz4,
    #[cfg(feature = "value-compression-zstd")]
    Zstd,
}

impl ValuePayloadEncoding {
    #[cfg(any(feature = "value-compression-lz4", feature = "value-compression-zstd"))]
    const fn format_id(self) -> u32 {
        match self {
            Self::Raw => VALUE_PAYLOAD_ENCODING_RAW,
            #[cfg(feature = "value-compression-lz4")]
            Self::Lz4 => VALUE_PAYLOAD_ENCODING_LZ4,
            #[cfg(feature = "value-compression-zstd")]
            Self::Zstd => VALUE_PAYLOAD_ENCODING_ZSTD,
        }
    }
}

#[cfg(feature = "value-compression-lz4")]
fn encode_lz4_frame(
    raw_payload: &[u8],
    policy: ValuePayloadCompressionPolicy,
    out: &mut Vec<u8>,
) -> Result<ValuePayloadFrame, FormatError> {
    if !policy.should_try(raw_payload.len()) {
        let frame = ValuePayloadFrame::raw_with_header(raw_payload.len())?;
        frame.write_header(out);
        out.extend_from_slice(raw_payload);
        return Ok(frame);
    }

    let compressed = lz4_flex::block::compress(raw_payload);
    if !policy.should_keep_compressed(raw_payload.len(), compressed.len()) {
        let frame = ValuePayloadFrame::raw_with_header(raw_payload.len())?;
        frame.write_header(out);
        out.extend_from_slice(raw_payload);
        return Ok(frame);
    }

    let frame = ValuePayloadFrame::lz4(raw_payload.len(), compressed.len())?;
    frame.write_header(out);
    out.extend_from_slice(&compressed);
    Ok(frame)
}

#[cfg(feature = "value-compression-zstd")]
fn encode_zstd_frame(
    raw_payload: &[u8],
    policy: ValuePayloadCompressionPolicy,
    out: &mut Vec<u8>,
    compressor: Option<&mut zstd::bulk::Compressor<'static>>,
    scratch: &mut Vec<u8>,
) -> Result<ValuePayloadFrame, FormatError> {
    if !policy.should_try(raw_payload.len()) {
        let frame = ValuePayloadFrame::raw_with_header(raw_payload.len())?;
        frame.write_header(out);
        out.extend_from_slice(raw_payload);
        return Ok(frame);
    }

    let compressed_len = {
        let bound = zstd::zstd_safe::compress_bound(raw_payload.len());
        if scratch.len() < bound {
            scratch.resize(bound, 0);
        }
        compressor
            .ok_or_else(|| FormatError::limit("zstd compressor context"))?
            .compress_to_buffer(raw_payload, scratch)
            .map_err(|_| FormatError::limit("zstd payload"))?
    };
    if !policy.should_keep_compressed(raw_payload.len(), compressed_len) {
        let frame = ValuePayloadFrame::raw_with_header(raw_payload.len())?;
        frame.write_header(out);
        out.extend_from_slice(raw_payload);
        return Ok(frame);
    }

    let frame = ValuePayloadFrame::zstd(raw_payload.len(), compressed_len)?;
    frame.write_header(out);
    out.extend_from_slice(&scratch[..compressed_len]);
    Ok(frame)
}

#[cfg(any(feature = "value-compression-lz4", feature = "value-compression-zstd"))]
fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, CorruptionError> {
    let chunk = bytes
        .get(offset..offset + 4)
        .and_then(<[u8]>::first_chunk::<4>)
        .ok_or(CorruptionError::Block)?;
    Ok(u32::from_le_bytes(*chunk))
}

#[cfg(test)]
mod tests {
    use super::*;

    mod none {
        use super::*;

        #[test]
        fn stores_payload_without_header() {
            let compression = ValuePayloadCompressionKind::None;
            let frame = b"abc";
            let layout =
                ValuePayloadFrame::raw_without_header(frame.len()).expect("frame should encode");

            assert_eq!(layout.frame_len(), 3);
            assert_eq!(layout.encoded_range(), 0..3);
            assert!(layout.is_raw_borrowable());
            let mut decoder = ValuePayloadDecoder::new(compression);
            assert_eq!(
                compression
                    .decode_frame(&mut decoder, frame, 3)
                    .expect("frame should decode")
                    .as_slice(),
                b"abc"
            );
        }
    }

    #[cfg(any(feature = "value-compression-lz4", feature = "value-compression-zstd"))]
    mod framed {
        use super::*;

        #[test]
        fn rejects_raw_frame_whose_encoded_len_differs_from_raw_len() {
            let compression = framed_compression_kind();
            let mut frame = Vec::new();
            frame.extend_from_slice(&VALUE_PAYLOAD_ENCODING_RAW.to_le_bytes());
            frame.extend_from_slice(&4u32.to_le_bytes());
            frame.extend_from_slice(&3u32.to_le_bytes());
            frame.extend_from_slice(b"abc");

            let mut decoder = ValuePayloadDecoder::new(compression);
            assert!(matches!(
                compression.decode_frame(&mut decoder, &frame, 4),
                Err(CorruptionError::Block)
            ));
        }

        fn framed_compression_kind() -> ValuePayloadCompressionKind {
            #[cfg(feature = "value-compression-lz4")]
            {
                ValuePayloadCompressionKind::Lz4
            }
            #[cfg(all(
                not(feature = "value-compression-lz4"),
                feature = "value-compression-zstd"
            ))]
            {
                ValuePayloadCompressionKind::ZstdLevel1
            }
        }
    }

    #[cfg(feature = "value-compression-lz4")]
    mod lz4 {
        use super::*;

        #[test]
        fn small_payload_falls_back_to_raw_frame() {
            let mut frame = Vec::new();
            let compression = ValuePayloadCompressionKind::Lz4;
            let mut encoder = ValuePayloadEncoder::new(compression);
            let layout = encoder
                .encode_frame(b"abc", ValuePayloadCompressionPolicy::DEFAULT, &mut frame)
                .expect("frame should encode");

            assert_eq!(layout.frame_len(), VALUE_PAYLOAD_FRAME_HEADER_LEN + 3);
            assert!(layout.is_raw_borrowable());
            assert_eq!(&frame[VALUE_PAYLOAD_FRAME_HEADER_LEN..], b"abc");
        }

        #[test]
        fn compressible_payload_round_trips() {
            let raw = vec![7u8; DEFAULT_VALUE_PAYLOAD_COMPRESSION_MIN_TRY_LEN * 2];
            let mut frame = Vec::new();
            let compression = ValuePayloadCompressionKind::Lz4;
            let mut encoder = ValuePayloadEncoder::new(compression);
            let layout = encoder
                .encode_frame(&raw, ValuePayloadCompressionPolicy::DEFAULT, &mut frame)
                .expect("frame should encode");

            assert!(!layout.is_raw_borrowable());
            assert!(frame.len() < raw.len());
            let mut decoder = ValuePayloadDecoder::new(compression);
            let decoded = compression
                .decode_frame(&mut decoder, &frame, raw.len())
                .expect("frame should decode");
            assert_eq!(decoded.as_slice(), raw);
        }

        #[test]
        fn high_saved_threshold_keeps_raw_frame() {
            let raw = vec![7u8; DEFAULT_VALUE_PAYLOAD_COMPRESSION_MIN_TRY_LEN * 2];
            let policy = ValuePayloadCompressionPolicy::DEFAULT.with_min_saved_percent(100);
            let mut frame = Vec::new();
            let mut encoder = ValuePayloadEncoder::new(ValuePayloadCompressionKind::Lz4);
            let layout = encoder
                .encode_frame(&raw, policy, &mut frame)
                .expect("frame should encode");

            assert!(layout.is_raw_borrowable());
            assert_eq!(
                layout.frame_len(),
                VALUE_PAYLOAD_FRAME_HEADER_LEN + raw.len()
            );
        }

        #[test]
        fn high_min_try_len_skips_compression_attempt() {
            let raw = vec![7u8; DEFAULT_VALUE_PAYLOAD_COMPRESSION_MIN_TRY_LEN * 2];
            let policy = ValuePayloadCompressionPolicy::DEFAULT.with_min_try_len(raw.len() + 1);
            let mut frame = Vec::new();
            let mut encoder = ValuePayloadEncoder::new(ValuePayloadCompressionKind::Lz4);
            let layout = encoder
                .encode_frame(&raw, policy, &mut frame)
                .expect("frame should encode");

            assert!(layout.is_raw_borrowable());
            assert_eq!(
                layout.frame_len(),
                VALUE_PAYLOAD_FRAME_HEADER_LEN + raw.len()
            );
        }
    }

    #[cfg(feature = "value-compression-zstd")]
    mod zstd {
        use super::*;

        #[test]
        fn small_payload_falls_back_to_raw_frame() {
            let mut frame = Vec::new();
            let compression = ValuePayloadCompressionKind::ZstdLevel1;
            let mut encoder = ValuePayloadEncoder::new(compression);
            let layout = encoder
                .encode_frame(b"abc", ValuePayloadCompressionPolicy::DEFAULT, &mut frame)
                .expect("frame should encode");

            assert_eq!(layout.frame_len(), VALUE_PAYLOAD_FRAME_HEADER_LEN + 3);
            assert!(layout.is_raw_borrowable());
            assert_eq!(&frame[VALUE_PAYLOAD_FRAME_HEADER_LEN..], b"abc");
        }

        #[test]
        fn compressible_payload_round_trips() {
            let raw = vec![7u8; DEFAULT_VALUE_PAYLOAD_COMPRESSION_MIN_TRY_LEN * 2];
            let mut frame = Vec::new();
            let compression = ValuePayloadCompressionKind::ZstdLevel1;
            let mut encoder = ValuePayloadEncoder::new(compression);
            let layout = encoder
                .encode_frame(&raw, ValuePayloadCompressionPolicy::DEFAULT, &mut frame)
                .expect("frame should encode");

            assert!(!layout.is_raw_borrowable());
            assert!(frame.len() < raw.len());
            let mut decoder = ValuePayloadDecoder::new(compression);
            let decoded = compression
                .decode_frame(&mut decoder, &frame, raw.len())
                .expect("frame should decode");
            assert_eq!(decoded.as_slice(), raw);
        }
    }

    impl DecodedPayload<'_> {
        fn as_slice(&self) -> &[u8] {
            match self {
                Self::Borrowed(bytes) => bytes,
                #[cfg(any(feature = "value-compression-lz4", feature = "value-compression-zstd"))]
                Self::Owned(bytes) => bytes,
            }
        }
    }
}
