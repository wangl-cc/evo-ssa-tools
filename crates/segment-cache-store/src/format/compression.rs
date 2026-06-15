//! Value-payload compression formats and block-local payload frames.
//!
//! Compression is a physical block-level concern. Keys and value indexes stay
//! uncompressed so ordered lookup can validate and search metadata before
//! deciding whether the value payload needs to be read.

use crate::format::{CorruptionError, FormatError, format_u32};

/// Minimum raw payload length before the LZ4 writer attempts compression.
#[cfg(feature = "value-compression-lz4")]
const LZ4_MIN_TRY_LEN: usize = 64 * 1024;
/// Minimum saved fraction before a compressed frame is kept.
#[cfg(feature = "value-compression-lz4")]
const LZ4_MIN_SAVED_RATIO_DENOMINATOR: usize = 10;

#[cfg(feature = "value-compression-lz4")]
pub(crate) const VALUE_PAYLOAD_FRAME_HEADER_LEN: usize = 12;

#[cfg(feature = "value-compression-lz4")]
const VALUE_PAYLOAD_ENCODING_RAW: u32 = 0;
#[cfg(feature = "value-compression-lz4")]
const VALUE_PAYLOAD_ENCODING_LZ4: u32 = 1;

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
            _ => None,
        }
    }

    /// Human-readable compression name used in diagnostics and debug output.
    pub const fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            #[cfg(feature = "value-compression-lz4")]
            Self::Lz4 => "lz4",
        }
    }

    /// Stable persisted compression identifier.
    pub const fn format_id(self) -> u32 {
        match self {
            Self::None => 0,
            #[cfg(feature = "value-compression-lz4")]
            Self::Lz4 => 1,
        }
    }

    /// Bytes stored before the encoded payload for this policy.
    pub const fn frame_header_len(self) -> usize {
        match self {
            Self::None => 0,
            #[cfg(feature = "value-compression-lz4")]
            Self::Lz4 => VALUE_PAYLOAD_FRAME_HEADER_LEN,
        }
    }

    #[cfg(any(test, feature = "value-compression-lz4"))]
    pub(crate) fn encode_frame(
        self,
        raw_payload: &[u8],
        out: &mut Vec<u8>,
    ) -> Result<ValuePayloadFrame, FormatError> {
        match self {
            Self::None => {
                out.extend_from_slice(raw_payload);
                Ok(ValuePayloadFrame::raw_without_header(raw_payload.len())?)
            }
            #[cfg(feature = "value-compression-lz4")]
            Self::Lz4 => encode_lz4_frame(raw_payload, out),
        }
    }

    pub(crate) fn parse_frame_header(
        self,
        _bytes: &[u8],
        expected_raw_len: usize,
    ) -> Result<ValuePayloadFrame, CorruptionError> {
        match self {
            Self::None => Ok(ValuePayloadFrame::raw_without_header(expected_raw_len)
                .map_err(|_| CorruptionError::Block)?),
            #[cfg(feature = "value-compression-lz4")]
            Self::Lz4 => ValuePayloadFrame::from_bytes(_bytes, expected_raw_len),
        }
    }

    pub(crate) fn decode_frame(
        self,
        frame: &[u8],
        expected_raw_len: usize,
    ) -> Result<DecodedPayload<'_>, CorruptionError> {
        let header = self.parse_frame_header(frame, expected_raw_len)?;
        let encoded = frame
            .get(header.encoded_range())
            .ok_or(CorruptionError::Block)?;
        match header.encoding {
            ValuePayloadEncoding::Raw => Ok(DecodedPayload::Borrowed(encoded)),
            #[cfg(feature = "value-compression-lz4")]
            ValuePayloadEncoding::Lz4 => {
                let mut decoded = vec![0u8; expected_raw_len];
                let written = lz4_flex::block::decompress_into(encoded, &mut decoded)
                    .map_err(|_| CorruptionError::Block)?;
                if written != expected_raw_len {
                    return Err(CorruptionError::Block);
                }
                Ok(DecodedPayload::Owned(decoded))
            }
        }
    }
}

pub(crate) enum DecodedPayload<'a> {
    Borrowed(&'a [u8]),
    #[cfg(feature = "value-compression-lz4")]
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

    #[cfg(feature = "value-compression-lz4")]
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

    #[cfg(feature = "value-compression-lz4")]
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
            _ => return Err(CorruptionError::Block),
        };
        Ok(Self {
            encoding,
            raw_len,
            encoded_len,
            header_len: VALUE_PAYLOAD_FRAME_HEADER_LEN,
        })
    }

    #[cfg(feature = "value-compression-lz4")]
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
}

impl ValuePayloadEncoding {
    #[cfg(feature = "value-compression-lz4")]
    const fn format_id(self) -> u32 {
        match self {
            Self::Raw => VALUE_PAYLOAD_ENCODING_RAW,
            #[cfg(feature = "value-compression-lz4")]
            Self::Lz4 => VALUE_PAYLOAD_ENCODING_LZ4,
        }
    }
}

#[cfg(feature = "value-compression-lz4")]
fn encode_lz4_frame(
    raw_payload: &[u8],
    out: &mut Vec<u8>,
) -> Result<ValuePayloadFrame, FormatError> {
    if raw_payload.len() < LZ4_MIN_TRY_LEN {
        let frame = ValuePayloadFrame::raw_with_header(raw_payload.len())?;
        frame.write_header(out);
        out.extend_from_slice(raw_payload);
        return Ok(frame);
    }

    let compressed = lz4_flex::block::compress(raw_payload);
    let saved = raw_payload.len().saturating_sub(compressed.len());
    if compressed.len() >= raw_payload.len()
        || saved.saturating_mul(LZ4_MIN_SAVED_RATIO_DENOMINATOR) < raw_payload.len()
    {
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

#[cfg(feature = "value-compression-lz4")]
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
            let mut frame = Vec::new();
            let layout = ValuePayloadCompressionKind::None
                .encode_frame(b"abc", &mut frame)
                .expect("frame should encode");

            assert_eq!(frame, b"abc");
            assert_eq!(layout.frame_len(), 3);
            assert_eq!(layout.encoded_range(), 0..3);
            assert!(layout.is_raw_borrowable());
            assert_eq!(
                ValuePayloadCompressionKind::None
                    .decode_frame(&frame, 3)
                    .expect("frame should decode")
                    .as_slice(),
                b"abc"
            );
        }
    }

    #[cfg(feature = "value-compression-lz4")]
    mod lz4 {
        use super::*;

        #[test]
        fn small_payload_falls_back_to_raw_frame() {
            let mut frame = Vec::new();
            let layout = ValuePayloadCompressionKind::Lz4
                .encode_frame(b"abc", &mut frame)
                .expect("frame should encode");

            assert_eq!(layout.frame_len(), VALUE_PAYLOAD_FRAME_HEADER_LEN + 3);
            assert!(layout.is_raw_borrowable());
            assert_eq!(&frame[VALUE_PAYLOAD_FRAME_HEADER_LEN..], b"abc");
        }

        #[test]
        fn compressible_payload_round_trips() {
            let raw = vec![7u8; LZ4_MIN_TRY_LEN * 2];
            let mut frame = Vec::new();
            let layout = ValuePayloadCompressionKind::Lz4
                .encode_frame(&raw, &mut frame)
                .expect("frame should encode");

            assert!(!layout.is_raw_borrowable());
            assert!(frame.len() < raw.len());
            let decoded = ValuePayloadCompressionKind::Lz4
                .decode_frame(&frame, raw.len())
                .expect("frame should decode");
            assert_eq!(decoded.as_slice(), raw);
        }
    }

    impl DecodedPayload<'_> {
        fn as_slice(&self) -> &[u8] {
            match self {
                Self::Borrowed(bytes) => bytes,
                #[cfg(feature = "value-compression-lz4")]
                Self::Owned(bytes) => bytes,
            }
        }
    }
}
