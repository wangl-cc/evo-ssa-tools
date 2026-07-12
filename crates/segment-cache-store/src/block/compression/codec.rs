//! Value-payload encoder and decoder.
//!
//! This module is compiled only when the `value-compression` capability is
//! enabled. Concrete backend features add their encoder and decoder states;
//! the `None` state keeps marker-only builds valid and writes raw payload bytes
//! without an intermediate allocation.

#[cfg(any(feature = "value-compression-lz4", feature = "value-compression-zstd"))]
use super::frame;
use super::{
    ValuePayloadCompressionKind, ValuePayloadCompressionPolicy,
    frame::{PayloadFrame, PayloadStorage},
};
use crate::error::{CorruptionError, FormatError};

pub(crate) struct ValuePayloadEncoder {
    state: EncoderState,
}

enum EncoderState {
    None,
    #[cfg(feature = "value-compression-lz4")]
    Lz4,
    #[cfg(feature = "value-compression-zstd")]
    Zstd {
        compressor: zstd::bulk::Compressor<'static>,
        scratch: Vec<u8>,
    },
}

pub(crate) struct ValuePayloadDecoder {
    state: DecoderState,
}

enum DecoderState {
    None,
    #[cfg(feature = "value-compression-lz4")]
    Lz4 {
        buffer: Vec<u8>,
    },
    #[cfg(feature = "value-compression-zstd")]
    Zstd {
        decompressor: zstd::bulk::Decompressor<'static>,
        buffer: Vec<u8>,
    },
}

impl ValuePayloadCompressionKind {
    /// Bytes stored before the encoded payload for this kind.
    pub(crate) fn frame_header_len(self) -> usize {
        match self {
            Self::None => 0,
            #[cfg(feature = "value-compression-lz4")]
            Self::Lz4 => frame::HEADER_LEN,
            #[cfg(feature = "value-compression-zstd")]
            Self::ZstdLevel1 => frame::HEADER_LEN,
        }
    }

    pub(crate) fn parse_frame(
        self,
        _bytes: &[u8],
        expected_raw_len: usize,
    ) -> Result<PayloadFrame, CorruptionError> {
        match self {
            Self::None => {
                if _bytes.len() != expected_raw_len {
                    return Err(CorruptionError::Block);
                }
                Ok(PayloadFrame::raw_without_header(expected_raw_len))
            }
            #[cfg(feature = "value-compression-lz4")]
            Self::Lz4 => frame::parse(_bytes, expected_raw_len),
            #[cfg(feature = "value-compression-zstd")]
            Self::ZstdLevel1 => frame::parse(_bytes, expected_raw_len),
        }
    }

    pub(crate) fn decode_frame(
        self,
        decoder: &mut ValuePayloadDecoder,
        frame: &[u8],
        expected_raw_len: usize,
    ) -> Result<Option<Vec<u8>>, CorruptionError> {
        decoder.decode_frame(self, frame, expected_raw_len)
    }
}

impl ValuePayloadEncoder {
    pub(crate) fn new(kind: ValuePayloadCompressionKind) -> Self {
        let state = match kind {
            ValuePayloadCompressionKind::None => EncoderState::None,
            #[cfg(feature = "value-compression-lz4")]
            ValuePayloadCompressionKind::Lz4 => EncoderState::Lz4,
            #[cfg(feature = "value-compression-zstd")]
            ValuePayloadCompressionKind::ZstdLevel1 => EncoderState::Zstd {
                compressor: zstd::bulk::Compressor::new(1)
                    .expect("zstd level 1 compressor context should be constructible"),
                scratch: Vec::new(),
            },
        };
        Self { state }
    }

    pub(crate) fn encode_frame<F>(
        &mut self,
        raw_len: usize,
        _policy: ValuePayloadCompressionPolicy,
        out: &mut Vec<u8>,
        write_raw: F,
    ) -> Result<PayloadFrame, FormatError>
    where
        F: FnOnce(&mut Vec<u8>),
    {
        match &mut self.state {
            EncoderState::None => {
                let start = out.len();
                write_raw(out);
                validate_written_len(out.len() - start, raw_len)?;
                Ok(PayloadFrame::raw_without_header(raw_len))
            }
            #[cfg(feature = "value-compression-lz4")]
            EncoderState::Lz4 => {
                let mut raw_payload = Vec::with_capacity(raw_len);
                write_raw(&mut raw_payload);
                validate_written_len(raw_payload.len(), raw_len)?;
                frame::encode_lz4(&raw_payload, _policy, out)
            }
            #[cfg(feature = "value-compression-zstd")]
            EncoderState::Zstd {
                compressor,
                scratch,
            } => {
                let mut raw_payload = Vec::with_capacity(raw_len);
                write_raw(&mut raw_payload);
                validate_written_len(raw_payload.len(), raw_len)?;
                frame::encode_zstd(&raw_payload, _policy, out, compressor, scratch)
            }
        }
    }
}

impl ValuePayloadDecoder {
    pub(crate) fn new(kind: ValuePayloadCompressionKind) -> Self {
        let state = match kind {
            ValuePayloadCompressionKind::None => DecoderState::None,
            #[cfg(feature = "value-compression-lz4")]
            ValuePayloadCompressionKind::Lz4 => DecoderState::Lz4 { buffer: Vec::new() },
            #[cfg(feature = "value-compression-zstd")]
            ValuePayloadCompressionKind::ZstdLevel1 => DecoderState::Zstd {
                decompressor: zstd::bulk::Decompressor::new()
                    .expect("zstd decompressor context should be constructible"),
                buffer: Vec::new(),
            },
        };
        Self { state }
    }

    pub(crate) fn reclaim_payload_buffer(&mut self, _buffer: Option<Vec<u8>>) {
        let Some(_buffer) = _buffer else {
            return;
        };
        match &mut self.state {
            DecoderState::None => {}
            #[cfg(feature = "value-compression-lz4")]
            DecoderState::Lz4 { buffer: current } => {
                if _buffer.capacity() > current.capacity() {
                    *current = _buffer;
                }
            }
            #[cfg(feature = "value-compression-zstd")]
            DecoderState::Zstd {
                buffer: current, ..
            } => {
                if _buffer.capacity() > current.capacity() {
                    *current = _buffer;
                }
            }
        }
    }

    fn decode_frame(
        &mut self,
        kind: ValuePayloadCompressionKind,
        frame: &[u8],
        expected_raw_len: usize,
    ) -> Result<Option<Vec<u8>>, CorruptionError> {
        let header = kind.parse_frame(frame, expected_raw_len)?;
        let encoded = frame
            .get(header.payload_range())
            .ok_or(CorruptionError::Block)?;
        match (header.storage(), &mut self.state) {
            (PayloadStorage::Raw, _) => {
                if encoded.len() != expected_raw_len {
                    return Err(CorruptionError::Block);
                }
                Ok(None)
            }
            #[cfg(feature = "value-compression-lz4")]
            (PayloadStorage::Compressed, DecoderState::Lz4 { buffer }) => {
                buffer.resize(expected_raw_len, 0);
                let written = lz4_flex::block::decompress_into(encoded, buffer)
                    .map_err(|_| CorruptionError::Block)?;
                if written != expected_raw_len {
                    return Err(CorruptionError::Block);
                }
                Ok(Some(std::mem::take(buffer)))
            }
            #[cfg(feature = "value-compression-zstd")]
            (
                PayloadStorage::Compressed,
                DecoderState::Zstd {
                    decompressor,
                    buffer,
                },
            ) => {
                buffer.resize(expected_raw_len, 0);
                let written = decompressor
                    .decompress_to_buffer(encoded, buffer)
                    .map_err(|_| CorruptionError::Block)?;
                if written != expected_raw_len {
                    return Err(CorruptionError::Block);
                }
                Ok(Some(std::mem::take(buffer)))
            }
            #[cfg(any(feature = "value-compression-lz4", feature = "value-compression-zstd"))]
            _ => Err(CorruptionError::Block),
        }
    }
}

fn validate_written_len(actual: usize, expected: usize) -> Result<(), FormatError> {
    if actual != expected {
        return Err(FormatError::limit("block value payload length"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_stores_payload_without_header_or_allocation() {
        let compression = ValuePayloadCompressionKind::None;
        let mut bytes = Vec::new();
        let mut encoder = ValuePayloadEncoder::new(compression);
        let frame = encoder
            .encode_frame(
                3,
                ValuePayloadCompressionPolicy::DEFAULT,
                &mut bytes,
                |raw| raw.extend_from_slice(b"abc"),
            )
            .expect("frame should encode");

        assert_eq!(bytes, b"abc");
        assert_eq!(frame.frame_len(), 3);
        assert_eq!(frame.payload_range(), 0..3);
        assert!(frame.is_raw_borrowable());
        let mut decoder = ValuePayloadDecoder::new(compression);
        assert_eq!(
            compression
                .decode_frame(&mut decoder, &bytes, 3)
                .expect("frame should decode"),
            None
        );
    }

    #[cfg(feature = "value-compression-lz4")]
    mod lz4 {
        use super::*;

        #[test]
        fn small_payload_falls_back_to_raw_frame() {
            let (frame, bytes) = encode_payload(
                ValuePayloadCompressionKind::Lz4,
                b"abc",
                ValuePayloadCompressionPolicy::DEFAULT,
            );

            assert_eq!(frame.frame_len(), frame::HEADER_LEN + 3);
            assert!(frame.is_raw_borrowable());
            assert_eq!(bytes[0], 0);
            assert_eq!(&bytes[frame::HEADER_LEN..], b"abc");
        }

        #[test]
        fn compressible_payload_round_trips() {
            let raw = vec![7u8; ValuePayloadCompressionPolicy::DEFAULT.min_try_len() * 2];
            let compression = ValuePayloadCompressionKind::Lz4;
            let (frame, bytes) =
                encode_payload(compression, &raw, ValuePayloadCompressionPolicy::DEFAULT);

            assert!(!frame.is_raw_borrowable());
            assert_eq!(bytes[0], 1);
            assert!(bytes.len() < raw.len());
            let mut decoder = ValuePayloadDecoder::new(compression);
            let decoded = compression
                .decode_frame(&mut decoder, &bytes, raw.len())
                .expect("frame should decode")
                .expect("payload should be compressed");
            assert_eq!(decoded, raw);
        }

        #[test]
        fn oversized_compressed_payload_is_rejected_before_allocation() {
            let compression = ValuePayloadCompressionKind::Lz4;
            let raw_len = frame::MAX_DECODED_PAYLOAD_LEN + 1;
            let bytes = vec![1];
            let mut decoder = ValuePayloadDecoder::new(compression);

            assert!(matches!(
                compression.decode_frame(&mut decoder, &bytes, raw_len),
                Err(CorruptionError::Block)
            ));
            let DecoderState::Lz4 { buffer } = decoder.state else {
                panic!("lz4 decoder state should be selected");
            };
            assert_eq!(buffer.capacity(), 0);
        }

        #[test]
        fn strict_policy_keeps_raw_frame() {
            let raw = vec![7u8; ValuePayloadCompressionPolicy::DEFAULT.min_try_len() * 2];
            let policy = ValuePayloadCompressionPolicy::DEFAULT
                .with_min_saved_percent(100)
                .expect("saved percentage is valid");
            let (frame, _) = encode_payload(ValuePayloadCompressionKind::Lz4, &raw, policy);
            assert!(frame.is_raw_borrowable());

            let policy = ValuePayloadCompressionPolicy::DEFAULT.with_min_try_len(raw.len() + 1);
            let (frame, _) = encode_payload(ValuePayloadCompressionKind::Lz4, &raw, policy);
            assert!(frame.is_raw_borrowable());
        }
    }

    #[cfg(feature = "value-compression-zstd")]
    mod zstd {
        use super::*;

        #[test]
        fn small_payload_falls_back_to_raw_frame() {
            let (frame, bytes) = encode_payload(
                ValuePayloadCompressionKind::ZstdLevel1,
                b"abc",
                ValuePayloadCompressionPolicy::DEFAULT,
            );

            assert_eq!(frame.frame_len(), frame::HEADER_LEN + 3);
            assert!(frame.is_raw_borrowable());
            assert_eq!(bytes[0], 0);
            assert_eq!(&bytes[frame::HEADER_LEN..], b"abc");
        }

        #[test]
        fn compressible_payload_round_trips() {
            let raw = vec![7u8; ValuePayloadCompressionPolicy::DEFAULT.min_try_len() * 2];
            let compression = ValuePayloadCompressionKind::ZstdLevel1;
            let (frame, bytes) =
                encode_payload(compression, &raw, ValuePayloadCompressionPolicy::DEFAULT);

            assert!(!frame.is_raw_borrowable());
            assert_eq!(bytes[0], 1);
            assert!(bytes.len() < raw.len());
            let mut decoder = ValuePayloadDecoder::new(compression);
            let decoded = compression
                .decode_frame(&mut decoder, &bytes, raw.len())
                .expect("frame should decode")
                .expect("payload should be compressed");
            assert_eq!(decoded, raw);
        }

        #[test]
        fn oversized_compressed_payload_is_rejected_before_allocation() {
            let compression = ValuePayloadCompressionKind::ZstdLevel1;
            let raw_len = frame::MAX_DECODED_PAYLOAD_LEN + 1;
            let bytes = vec![1];
            let mut decoder = ValuePayloadDecoder::new(compression);

            assert!(matches!(
                compression.decode_frame(&mut decoder, &bytes, raw_len),
                Err(CorruptionError::Block)
            ));
            let DecoderState::Zstd { buffer, .. } = decoder.state else {
                panic!("zstd decoder state should be selected");
            };
            assert_eq!(buffer.capacity(), 0);
        }
    }

    #[cfg(any(feature = "value-compression-lz4", feature = "value-compression-zstd"))]
    fn encode_payload(
        compression: ValuePayloadCompressionKind,
        raw: &[u8],
        policy: ValuePayloadCompressionPolicy,
    ) -> (PayloadFrame, Vec<u8>) {
        let mut bytes = Vec::new();
        let mut encoder = ValuePayloadEncoder::new(compression);
        let frame = encoder
            .encode_frame(raw.len(), policy, &mut bytes, |payload| {
                payload.extend_from_slice(raw);
            })
            .expect("frame should encode");
        (frame, bytes)
    }
}
