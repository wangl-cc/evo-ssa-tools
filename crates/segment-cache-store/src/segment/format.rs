//! Segment header, footer, and sparse block-index codecs.
//!
//! A segment is an immutable sorted file. The fixed-size header identifies
//! format and store geometry; the variable-size footer at the end of the file
//! is the completion marker and owns the sparse block index. Reading segment
//! bytes from disk lives in [`super::file`]; this module only encodes and decodes
//! them.

use crc32c::crc32c;

use super::index::{BlockIndexEntry, BlockKeyRange};
use crate::{
    binary::{BinaryCursor, format_u32},
    block::{BLOCK_METADATA_HEADER_LEN, BlockChecksumKind, ValuePayloadCompressionKind},
    error::{CorruptionError, FormatError},
    value::ValueLayout,
};

const SEGMENT_FORMAT_VERSION: u8 = 1;
const SEGMENT_HEADER_BODY_LEN: usize = 16;
pub(crate) const SEGMENT_HEADER_LEN: usize = 20;
pub(crate) const SEGMENT_FOOTER_TRAILER_LEN: usize = 8;
const HEADER_MAGIC: &[u8; 4] = b"SCSG";

// Segment header.

/// Parsed fixed-size segment header.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SegmentHeader {
    version: u8,
    key_len: u32,
    value_layout: ValueLayout,
    block_checksum_id: u8,
    value_payload_compression_id: u8,
}

impl SegmentHeader {
    pub(crate) fn new(
        key_len: usize,
        value_layout: ValueLayout,
        block_checksum: BlockChecksumKind,
        value_payload_compression: ValuePayloadCompressionKind,
    ) -> Result<Self, FormatError> {
        Ok(Self {
            version: SEGMENT_FORMAT_VERSION,
            key_len: u32::try_from(key_len)
                .map_err(|_| FormatError::limit("segment key length"))?,
            value_layout,
            block_checksum_id: block_checksum.format_id(),
            value_payload_compression_id: value_payload_compression.format_id(),
        })
    }

    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self, CorruptionError> {
        if bytes.len() != SEGMENT_HEADER_LEN || &bytes[..4] != HEADER_MAGIC {
            return Err(CorruptionError::SegmentFormat);
        }
        let stored_crc = u32::from_le_bytes(
            bytes[SEGMENT_HEADER_BODY_LEN..SEGMENT_HEADER_LEN]
                .try_into()
                .map_err(|_| CorruptionError::SegmentFormat)?,
        );
        if crc32c(&bytes[..SEGMENT_HEADER_BODY_LEN]) != stored_crc {
            return Err(CorruptionError::SegmentFormat);
        }
        let version = bytes[4];
        let block_checksum_id = bytes[5];
        let value_payload_compression_id = bytes[6];
        if bytes[7] != 0 {
            return Err(CorruptionError::SegmentFormat);
        }
        let mut cursor = BinaryCursor::at(bytes, 8);
        let key_len = cursor.read::<u32>().ok_or(CorruptionError::SegmentFormat)?;
        let value_len = cursor.read::<u32>().ok_or(CorruptionError::SegmentFormat)?;
        Ok(Self {
            version,
            key_len,
            value_layout: ValueLayout::from_u32(value_len),
            block_checksum_id,
            value_payload_compression_id,
        })
    }

    pub(crate) fn into_bytes(self, buffer: &mut Vec<u8>) {
        let start = buffer.len();
        buffer.extend_from_slice(HEADER_MAGIC);
        buffer.extend_from_slice(&[
            self.version,
            self.block_checksum_id,
            self.value_payload_compression_id,
            0,
        ]);
        buffer.extend_from_slice(&self.key_len.to_le_bytes());
        buffer.extend_from_slice(&self.value_layout.to_u32().to_le_bytes());
        let crc = crc32c(&buffer[start..start + SEGMENT_HEADER_BODY_LEN]);
        buffer.extend_from_slice(&crc.to_le_bytes());
    }

    pub(crate) fn matches_geometry(
        self,
        expected_key_len: usize,
        expected_value_layout: ValueLayout,
        expected_block_checksum: BlockChecksumKind,
        expected_value_payload_compression: ValuePayloadCompressionKind,
    ) -> bool {
        self.version == SEGMENT_FORMAT_VERSION
            && self.key_len as usize == expected_key_len
            && self.value_layout == expected_value_layout
            && self.block_checksum_id == expected_block_checksum.format_id()
            && self.value_payload_compression_id == expected_value_payload_compression.format_id()
    }
}

// Segment footer.

/// Parsed segment footer.
///
/// The footer is the completion gate for a segment and owns the block index.
#[derive(Clone, Debug)]
pub(crate) struct SegmentFooter {
    pub(crate) record_count: u64,
    pub(crate) block_index: Vec<BlockIndexEntry>,
}

impl SegmentFooter {
    pub(crate) fn write_to(&self, key_len: usize, buffer: &mut Vec<u8>) -> Result<(), FormatError> {
        let footer_body_start = buffer.len();
        buffer.extend_from_slice(&self.record_count.to_le_bytes());
        BlockIndexCodec::new(key_len).write_to(&self.block_index, buffer)?;
        let footer_body_len = format_u32(
            buffer.len() - footer_body_start,
            "segment footer body length",
        )?;
        buffer.extend_from_slice(&footer_body_len.to_le_bytes());
        let crc = crc32c(&buffer[footer_body_start..]);
        buffer.extend_from_slice(&crc.to_le_bytes());
        Ok(())
    }

    /// Decodes a footer from the final `footer_body_len + 8` bytes of a
    /// segment file. `data_end` is the absolute file offset where the footer
    /// starts (the end of the data blocks), used to validate the block index.
    pub(crate) fn from_bytes(
        bytes: &[u8],
        key_len: usize,
        data_end: u64,
        expected_min_key: &[u8],
        expected_max_key: &[u8],
    ) -> Result<Self, CorruptionError> {
        if bytes.len() < SEGMENT_FOOTER_TRAILER_LEN {
            return Err(CorruptionError::SegmentFormat);
        }
        let footer_body_len_offset = bytes.len() - SEGMENT_FOOTER_TRAILER_LEN;
        let crc_offset = footer_body_len_offset + 4;
        let mut trailer_cursor = BinaryCursor::at(bytes, footer_body_len_offset);
        let footer_body_len = trailer_cursor
            .read::<u32>()
            .ok_or(CorruptionError::SegmentFormat)? as usize;
        if footer_body_len != footer_body_len_offset {
            return Err(CorruptionError::SegmentFormat);
        }
        trailer_cursor.seek(crc_offset);
        let stored_crc = trailer_cursor
            .read::<u32>()
            .ok_or(CorruptionError::SegmentFormat)?;
        if crc32c(&bytes[..crc_offset]) != stored_crc {
            return Err(CorruptionError::SegmentFormat);
        }

        let footer_body = &bytes[..footer_body_len];
        let mut cursor = BinaryCursor::new(footer_body);
        let record_count = cursor.read::<u64>().ok_or(CorruptionError::SegmentFormat)?;
        let codec = BlockIndexCodec::new(key_len);
        let block_index = codec.read_entries(&mut cursor, record_count)?;
        let block_index =
            codec.validate_entries(block_index, expected_min_key, expected_max_key, data_end)?;
        Ok(Self {
            record_count,
            block_index,
        })
    }
}

// Block index entry and codec.

struct RawBlockIndexEntry {
    key_range: BlockKeyRange,
    block_offset: u64,
}

struct BlockIndexCodec {
    key_len: usize,
}

impl BlockIndexCodec {
    fn new(key_len: usize) -> Self {
        Self { key_len }
    }

    fn write_to(
        &self,
        entries: &[BlockIndexEntry],
        buffer: &mut Vec<u8>,
    ) -> Result<(), FormatError> {
        buffer.extend_from_slice(
            &format_u32(entries.len(), "block index entry count")?.to_le_bytes(),
        );
        for entry in entries {
            buffer.extend_from_slice(
                &format_u32(entry.key_range.prefix().len(), "block key prefix length")?
                    .to_le_bytes(),
            );
            buffer.extend_from_slice(entry.key_range.prefix());
            buffer.extend_from_slice(entry.key_range.min_suffix());
            buffer.extend_from_slice(entry.key_range.max_suffix());
            buffer.extend_from_slice(&entry.byte_range.start.to_le_bytes());
        }
        Ok(())
    }

    fn read_entries(
        &self,
        cursor: &mut BinaryCursor<'_>,
        record_count: u64,
    ) -> Result<Vec<RawBlockIndexEntry>, CorruptionError> {
        let count = cursor.read::<u32>().ok_or(CorruptionError::SegmentFormat)?;
        let count = count as usize;
        let minimum_entry_len = self
            .key_len
            .checked_add(4 + 8)
            .ok_or(CorruptionError::SegmentFormat)?;
        if count == 0
            || record_count == 0
            || u64::try_from(count).map_err(|_| CorruptionError::SegmentFormat)? > record_count
            || count
                .checked_mul(minimum_entry_len)
                .is_none_or(|minimum_len| minimum_len > cursor.remaining())
        {
            return Err(CorruptionError::SegmentFormat);
        }
        let mut entries = Vec::with_capacity(count);
        for _ in 0..count {
            let prefix_len = cursor.read::<u32>().ok_or(CorruptionError::SegmentFormat)? as usize;
            let suffix_len = self
                .key_len
                .checked_sub(prefix_len)
                .ok_or(CorruptionError::SegmentFormat)?;
            let encoded_range_len = prefix_len
                .checked_add(
                    suffix_len
                        .checked_mul(2)
                        .ok_or(CorruptionError::SegmentFormat)?,
                )
                .ok_or(CorruptionError::SegmentFormat)?;
            let encoded_range = cursor
                .read_vec(encoded_range_len)
                .ok_or(CorruptionError::SegmentFormat)?;
            let block_offset = cursor.read::<u64>().ok_or(CorruptionError::SegmentFormat)?;
            entries.push(RawBlockIndexEntry {
                key_range: BlockKeyRange::from_encoded(encoded_range, prefix_len, self.key_len)
                    .ok_or(CorruptionError::SegmentFormat)?,
                block_offset,
            });
        }
        if cursor.remaining() != 0 {
            return Err(CorruptionError::SegmentFormat);
        }
        Ok(entries)
    }

    fn validate_entries(
        &self,
        entries: Vec<RawBlockIndexEntry>,
        expected_min_key: &[u8],
        expected_max_key: &[u8],
        data_end: u64,
    ) -> Result<Vec<BlockIndexEntry>, CorruptionError> {
        if expected_min_key.len() != self.key_len || expected_max_key.len() != self.key_len {
            return Err(CorruptionError::SegmentFormat);
        }
        if !entries[0].key_range.min_equals(expected_min_key)
            || !entries
                .last()
                .is_some_and(|entry| entry.key_range.max_equals(expected_max_key))
        {
            return Err(CorruptionError::SegmentFormat);
        }

        let mut validated: Vec<BlockIndexEntry> = Vec::with_capacity(entries.len());
        let mut block_end = data_end;
        for entry in entries.into_iter().rev() {
            if entry.block_offset < SEGMENT_HEADER_LEN as u64
                || block_end <= entry.block_offset
                || block_end - entry.block_offset < BLOCK_METADATA_HEADER_LEN as u64
            {
                return Err(CorruptionError::SegmentFormat);
            }
            validated.push(BlockIndexEntry {
                key_range: entry.key_range,
                byte_range: entry.block_offset..block_end,
            });
            block_end = entry.block_offset;
        }
        validated.reverse();
        if validated[0].byte_range.start != SEGMENT_HEADER_LEN as u64
            || validated.last().map(|entry| entry.byte_range.end) != Some(data_end)
            || validated
                .windows(2)
                .any(|entries| !entries[0].key_range.ends_before(&entries[1].key_range))
        {
            return Err(CorruptionError::SegmentFormat);
        }
        Ok(validated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::{BlockChecksumKind, ValuePayloadCompressionKind};

    mod header {
        use super::*;

        #[test]
        fn encodes_v1_field_order_and_widths() {
            let mut bytes = Vec::new();
            SegmentHeader::new(
                16,
                ValueLayout::fixed(std::num::NonZeroU32::new(32).expect("non-zero")),
                BlockChecksumKind::None,
                ValuePayloadCompressionKind::None,
            )
            .expect("header should encode")
            .into_bytes(&mut bytes);

            assert_eq!(bytes.len(), SEGMENT_HEADER_LEN);
            assert_eq!(&bytes[..4], HEADER_MAGIC);
            assert_eq!(bytes[4], SEGMENT_FORMAT_VERSION);
            assert_eq!(bytes[5], BlockChecksumKind::None.format_id());
            assert_eq!(bytes[6], ValuePayloadCompressionKind::None.format_id());
            assert_eq!(bytes[7], 0);
            assert_eq!(
                u32::from_le_bytes(bytes[8..12].try_into().expect("key len")),
                16
            );
            assert_eq!(
                u32::from_le_bytes(bytes[12..16].try_into().expect("value len")),
                32
            );
            assert_eq!(
                u32::from_le_bytes(bytes[16..20].try_into().expect("header crc")),
                crc32c(&bytes[..SEGMENT_HEADER_BODY_LEN])
            );
        }

        #[test]
        fn rejects_nonzero_reserved_byte() {
            let mut bytes = Vec::new();
            SegmentHeader::new(
                16,
                ValueLayout::VARIABLE,
                BlockChecksumKind::None,
                ValuePayloadCompressionKind::None,
            )
            .expect("header should encode")
            .into_bytes(&mut bytes);
            bytes[7] = 1;
            let crc = crc32c(&bytes[..SEGMENT_HEADER_BODY_LEN]);
            bytes[SEGMENT_HEADER_BODY_LEN..].copy_from_slice(&crc.to_le_bytes());

            assert!(matches!(
                SegmentHeader::from_bytes(&bytes),
                Err(CorruptionError::SegmentFormat)
            ));
        }
    }

    mod footer {
        use super::*;

        #[test]
        fn derives_block_ranges_from_offsets_and_footer_start() {
            let footer = SegmentFooter {
                record_count: 2,
                block_index: vec![
                    BlockIndexEntry {
                        key_range: BlockKeyRange::new(b"a0", b"a9").expect("valid range"),
                        byte_range: SEGMENT_HEADER_LEN as u64..40,
                    },
                    BlockIndexEntry {
                        key_range: BlockKeyRange::new(b"b0", b"b9").expect("valid range"),
                        byte_range: 40..64,
                    },
                ],
            };
            let mut bytes = Vec::new();
            footer
                .write_to(2, &mut bytes)
                .expect("footer should encode");
            assert_eq!(bytes.len(), 8 + 4 + 2 * (4 + 1 + 1 + 1 + 8) + 8);

            let decoded = SegmentFooter::from_bytes(&bytes, 2, 64, b"a0", b"b9")
                .expect("footer offsets should define valid ranges");
            assert_eq!(
                decoded.block_index[0].byte_range,
                SEGMENT_HEADER_LEN as u64..40
            );
            assert_eq!(decoded.block_index[1].byte_range, 40..64);
        }

        #[test]
        fn rejects_block_count_that_exceeds_record_count() {
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&1u64.to_le_bytes());
            bytes.extend_from_slice(&u32::MAX.to_le_bytes());
            let footer_body_len = u32::try_from(bytes.len()).expect("small test footer");
            bytes.extend_from_slice(&footer_body_len.to_le_bytes());
            let crc = crc32c(&bytes);
            bytes.extend_from_slice(&crc.to_le_bytes());

            assert!(matches!(
                SegmentFooter::from_bytes(&bytes, 2, SEGMENT_HEADER_LEN as u64, b"a0", b"a0"),
                Err(CorruptionError::SegmentFormat)
            ));
        }
    }
}
