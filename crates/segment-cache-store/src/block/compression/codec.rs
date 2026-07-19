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
use crate::{
    error::{CorruptionError, FormatError},
    limits::MAX_VALUE_PAYLOAD_LEN,
};

pub(crate) struct ValuePayloadEncoder {
    state: EncoderState,
}

enum EncoderState {
    None,
    #[cfg(feature = "value-compression-lz4")]
    Lz4 {
        table: Option<lz4_flex::block::CompressTable>,
        scratch: Vec<u8>,
    },
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
        if expected_raw_len > MAX_VALUE_PAYLOAD_LEN {
            return Err(CorruptionError::Block);
        }
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
            ValuePayloadCompressionKind::Lz4 => EncoderState::Lz4 {
                table: None,
                scratch: Vec::new(),
            },
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
        if raw_len > MAX_VALUE_PAYLOAD_LEN {
            return Err(FormatError::limit("block value payload length"));
        }
        match &mut self.state {
            EncoderState::None => {
                let start = out.len();
                write_raw(out);
                validate_written_len(out.len() - start, raw_len)?;
                Ok(PayloadFrame::raw_without_header(raw_len))
            }
            #[cfg(feature = "value-compression-lz4")]
            EncoderState::Lz4 { table, scratch } => {
                let pending = frame::PendingPayloadFrame::begin(out);
                write_raw(out);
                validate_written_len(pending.payload_len(out)?, raw_len)?;
                pending.finish_lz4(out, _policy, table, scratch)
            }
            #[cfg(feature = "value-compression-zstd")]
            EncoderState::Zstd {
                compressor,
                scratch,
            } => {
                let pending = frame::PendingPayloadFrame::begin(out);
                write_raw(out);
                validate_written_len(pending.payload_len(out)?, raw_len)?;
                pending.finish_zstd(out, _policy, compressor, scratch)
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
            let mut encoder = ValuePayloadEncoder::new(ValuePayloadCompressionKind::Lz4);
            let mut bytes = Vec::new();
            let frame = encoder
                .encode_frame(
                    3,
                    ValuePayloadCompressionPolicy::DEFAULT,
                    &mut bytes,
                    |payload| payload.extend_from_slice(b"abc"),
                )
                .expect("frame should encode");

            assert_eq!(frame.frame_len(), frame::HEADER_LEN + 3);
            assert!(frame.is_raw_borrowable());
            assert_eq!(bytes[0], 0);
            assert_eq!(&bytes[frame::HEADER_LEN..], b"abc");
            let EncoderState::Lz4 { table, scratch } = encoder.state else {
                panic!("lz4 encoder state should be selected");
            };
            assert!(table.is_none());
            assert_eq!(scratch.capacity(), 0);
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
            assert_eq!(&bytes[frame::HEADER_LEN..], lz4_flex::block::compress(&raw));
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
            let raw_len = MAX_VALUE_PAYLOAD_LEN + 1;
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
            let compression = ValuePayloadCompressionKind::Lz4;
            let (frame, bytes) = encode_payload(compression, &raw, policy);
            assert!(frame.is_raw_borrowable());
            assert_eq!(bytes[0], 0);
            assert_eq!(&bytes[frame::HEADER_LEN..], raw);
            let mut decoder = ValuePayloadDecoder::new(compression);
            assert_eq!(
                compression
                    .decode_frame(&mut decoder, &bytes, raw.len())
                    .expect("raw frame should decode"),
                None
            );

            let policy = ValuePayloadCompressionPolicy::DEFAULT.with_min_try_len(raw.len() + 1);
            let (frame, _) = encode_payload(ValuePayloadCompressionKind::Lz4, &raw, policy);
            assert!(frame.is_raw_borrowable());
        }

        #[test]
        fn reuses_compression_scratch_between_frames() {
            let raw = vec![7u8; ValuePayloadCompressionPolicy::DEFAULT.min_try_len() * 2];
            let mut encoder = ValuePayloadEncoder::new(ValuePayloadCompressionKind::Lz4);

            let mut first = Vec::new();
            encoder
                .encode_frame(
                    raw.len(),
                    ValuePayloadCompressionPolicy::DEFAULT,
                    &mut first,
                    |payload| payload.extend_from_slice(&raw),
                )
                .expect("first frame should encode");
            let EncoderState::Lz4 { scratch, .. } = &encoder.state else {
                panic!("lz4 encoder state should be selected");
            };
            let first_scratch = (scratch.as_ptr(), scratch.capacity());
            assert!(first_scratch.1 > 0);

            let mut second = Vec::new();
            encoder
                .encode_frame(
                    raw.len(),
                    ValuePayloadCompressionPolicy::DEFAULT,
                    &mut second,
                    |payload| payload.extend_from_slice(&raw),
                )
                .expect("second frame should encode");
            let EncoderState::Lz4 { scratch, .. } = &encoder.state else {
                panic!("lz4 encoder state should be selected");
            };
            assert_eq!((scratch.as_ptr(), scratch.capacity()), first_scratch);
            assert_eq!(second, first);
        }
    }

    #[cfg(feature = "value-compression-zstd")]
    mod zstd {
        use super::*;

        #[test]
        fn small_payload_falls_back_to_raw_frame() {
            let mut encoder = ValuePayloadEncoder::new(ValuePayloadCompressionKind::ZstdLevel1);
            let mut bytes = Vec::new();
            let frame = encoder
                .encode_frame(
                    3,
                    ValuePayloadCompressionPolicy::DEFAULT,
                    &mut bytes,
                    |payload| payload.extend_from_slice(b"abc"),
                )
                .expect("frame should encode");

            assert_eq!(frame.frame_len(), frame::HEADER_LEN + 3);
            assert!(frame.is_raw_borrowable());
            assert_eq!(bytes[0], 0);
            assert_eq!(&bytes[frame::HEADER_LEN..], b"abc");
            let EncoderState::Zstd { scratch, .. } = encoder.state else {
                panic!("zstd encoder state should be selected");
            };
            assert_eq!(scratch.capacity(), 0);
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
            assert_eq!(
                &bytes[frame::HEADER_LEN..],
                ::zstd::bulk::compress(&raw, 1).expect("reference payload should compress")
            );
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
            let raw_len = MAX_VALUE_PAYLOAD_LEN + 1;
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

        #[test]
        fn strict_policy_keeps_raw_frame() {
            let raw = vec![7u8; ValuePayloadCompressionPolicy::DEFAULT.min_try_len() * 2];
            let policy = ValuePayloadCompressionPolicy::DEFAULT
                .with_min_saved_percent(100)
                .expect("saved percentage is valid");
            let compression = ValuePayloadCompressionKind::ZstdLevel1;
            let (frame, bytes) = encode_payload(compression, &raw, policy);

            assert!(frame.is_raw_borrowable());
            assert_eq!(bytes[0], 0);
            assert_eq!(&bytes[frame::HEADER_LEN..], raw);
            let mut decoder = ValuePayloadDecoder::new(compression);
            assert_eq!(
                compression
                    .decode_frame(&mut decoder, &bytes, raw.len())
                    .expect("raw frame should decode"),
                None
            );
        }

        #[test]
        fn reuses_compression_scratch_between_frames() {
            let raw = vec![7u8; ValuePayloadCompressionPolicy::DEFAULT.min_try_len() * 2];
            let mut encoder = ValuePayloadEncoder::new(ValuePayloadCompressionKind::ZstdLevel1);

            let mut first = Vec::new();
            encoder
                .encode_frame(
                    raw.len(),
                    ValuePayloadCompressionPolicy::DEFAULT,
                    &mut first,
                    |payload| payload.extend_from_slice(&raw),
                )
                .expect("first frame should encode");
            let EncoderState::Zstd { scratch, .. } = &encoder.state else {
                panic!("zstd encoder state should be selected");
            };
            let first_scratch = (scratch.as_ptr(), scratch.capacity());
            assert!(first_scratch.1 > 0);

            let mut second = Vec::new();
            encoder
                .encode_frame(
                    raw.len(),
                    ValuePayloadCompressionPolicy::DEFAULT,
                    &mut second,
                    |payload| payload.extend_from_slice(&raw),
                )
                .expect("second frame should encode");
            let EncoderState::Zstd { scratch, .. } = &encoder.state else {
                panic!("zstd encoder state should be selected");
            };
            assert_eq!((scratch.as_ptr(), scratch.capacity()), first_scratch);
            assert_eq!(second, first);
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
