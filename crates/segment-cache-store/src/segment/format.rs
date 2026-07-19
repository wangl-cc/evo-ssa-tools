//! Segment header, footer, and sparse block-index codecs.
//!
//! A segment is an immutable sorted file. The fixed-size header identifies
//! format and store geometry; the variable-size footer at the end of the file
//! is the completion marker and owns the sparse block index. Reading segment
//! bytes from disk lives in [`super::file`]; this module only encodes and decodes
//! them.

use crc32c::crc32c;

use super::index::{EncodedBlockIndexEntry, SegmentIndex};
use crate::{
    binary::{BinaryCursor, format_u32},
    block::{BLOCK_METADATA_HEADER_LEN, BlockChecksumKind, ValuePayloadCompressionKind},
    error::{CorruptionError, FormatError},
    limits::{MAX_BLOCK_COUNT, MAX_FOOTER_LEN},
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
    pub(crate) block_index: SegmentIndex,
}

impl SegmentFooter {
    pub(crate) fn write_to(&self, key_len: usize, buffer: &mut Vec<u8>) -> Result<(), FormatError> {
        let footer_body_start = buffer.len();
        if self.block_index.len() > MAX_BLOCK_COUNT {
            return Err(FormatError::limit("segment block count"));
        }
        let codec = BlockIndexCodec::new(key_len);
        codec.write_to(&self.block_index, buffer)?;
        let footer_body_len = format_u32(
            buffer.len() - footer_body_start,
            "segment footer body length",
        )?;
        if buffer.len() - footer_body_start + SEGMENT_FOOTER_TRAILER_LEN > MAX_FOOTER_LEN {
            return Err(FormatError::limit("segment footer length"));
        }
        buffer.extend_from_slice(&footer_body_len.to_le_bytes());
        let crc = crc32c(&buffer[footer_body_start..]);
        buffer.extend_from_slice(&crc.to_le_bytes());
        Ok(())
    }

    /// Decodes a footer from the final `footer_body_len + 8` bytes of a
    /// segment file. `data_end` is the absolute file offset where the footer
    /// starts (the end of the data blocks), used to validate the block index.
    pub(crate) fn from_bytes(
        bytes: Vec<u8>,
        key_len: usize,
        data_end: u64,
        expected_min_key: &[u8],
        expected_max_key: &[u8],
    ) -> Result<Self, CorruptionError> {
        if bytes.len() < SEGMENT_FOOTER_TRAILER_LEN || bytes.len() > MAX_FOOTER_LEN {
            return Err(CorruptionError::SegmentFormat);
        }
        let footer_body_len_offset = bytes.len() - SEGMENT_FOOTER_TRAILER_LEN;
        let crc_offset = footer_body_len_offset + 4;
        let mut trailer_cursor = BinaryCursor::at(&bytes, footer_body_len_offset);
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

        let codec = BlockIndexCodec::new(key_len);
        let block_index = codec.read_and_validate(
            bytes,
            footer_body_len,
            expected_min_key,
            expected_max_key,
            data_end,
        )?;
        Ok(Self { block_index })
    }
}

// Block index entry and codec.

struct BlockIndexCodec {
    key_len: usize,
}

impl BlockIndexCodec {
    fn new(key_len: usize) -> Self {
        Self { key_len }
    }

    fn write_to(&self, index: &SegmentIndex, buffer: &mut Vec<u8>) -> Result<(), FormatError> {
        buffer.extend_from_slice(
            &format_u32(index.segment_prefix().len(), "segment key prefix length")?.to_le_bytes(),
        );
        buffer.extend_from_slice(index.segment_prefix().as_slice());
        buffer
            .extend_from_slice(&format_u32(index.len(), "block index entry count")?.to_le_bytes());
        for (entry, range) in index.entries() {
            buffer.extend_from_slice(
                &format_u32(range.extra_prefix.len(), "block extra key prefix length")?
                    .to_le_bytes(),
            );
            buffer.extend_from_slice(range.extra_prefix);
            buffer.extend_from_slice(range.min_suffix);
            buffer.extend_from_slice(range.max_suffix);
            buffer.extend_from_slice(&entry.byte_range.start.to_le_bytes());
        }
        Ok(())
    }

    fn read_and_validate(
        &self,
        bytes: Vec<u8>,
        footer_body_len: usize,
        expected_min_key: &[u8],
        expected_max_key: &[u8],
        data_end: u64,
    ) -> Result<SegmentIndex, CorruptionError> {
        if expected_min_key.len() != self.key_len || expected_max_key.len() != self.key_len {
            return Err(CorruptionError::SegmentFormat);
        }
        let mut cursor = BinaryCursor::new(&bytes[..footer_body_len]);
        let segment_prefix_len =
            cursor.read::<u32>().ok_or(CorruptionError::SegmentFormat)? as usize;
        if segment_prefix_len > self.key_len {
            return Err(CorruptionError::SegmentFormat);
        }
        let segment_prefix_start = cursor.position();
        cursor
            .read_slice(segment_prefix_len)
            .ok_or(CorruptionError::SegmentFormat)?;
        let segment_prefix = segment_prefix_start..cursor.position();
        let count = cursor.read::<u32>().ok_or(CorruptionError::SegmentFormat)?;
        let count = count as usize;
        let relative_key_len = self.key_len - segment_prefix_len;
        let minimum_entry_len = self
            .key_len
            .checked_sub(segment_prefix_len)
            .and_then(|len| len.checked_add(4 + 8))
            .ok_or(CorruptionError::SegmentFormat)?;
        if count == 0
            || count > MAX_BLOCK_COUNT
            || count
                .checked_mul(minimum_entry_len)
                .is_none_or(|minimum_len| minimum_len > cursor.remaining())
        {
            return Err(CorruptionError::SegmentFormat);
        }
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(count)
            .map_err(|_| CorruptionError::SegmentFormat)?;
        for _ in 0..count {
            let extra_prefix_len =
                cursor.read::<u32>().ok_or(CorruptionError::SegmentFormat)? as usize;
            let suffix_len = relative_key_len
                .checked_sub(extra_prefix_len)
                .ok_or(CorruptionError::SegmentFormat)?;
            let extra_prefix = Self::read_range(&mut cursor, extra_prefix_len)?;
            let min_suffix = Self::read_range(&mut cursor, suffix_len)?;
            let max_suffix = Self::read_range(&mut cursor, suffix_len)?;
            let block_offset = cursor.read::<u64>().ok_or(CorruptionError::SegmentFormat)?;
            entries.push((extra_prefix, min_suffix, max_suffix, block_offset));
        }
        if cursor.remaining() != 0 {
            return Err(CorruptionError::SegmentFormat);
        }
        let mut encoded_entries = Vec::new();
        encoded_entries
            .try_reserve_exact(entries.len())
            .map_err(|_| CorruptionError::SegmentFormat)?;
        for index in 0..entries.len() {
            let (extra_prefix, min_suffix, max_suffix, block_offset) = &entries[index];
            let block_end = entries.get(index + 1).map_or(data_end, |entry| entry.3);
            if *block_offset < SEGMENT_HEADER_LEN as u64
                || block_end <= *block_offset
                || block_end - *block_offset < BLOCK_METADATA_HEADER_LEN as u64
            {
                return Err(CorruptionError::SegmentFormat);
            }
            encoded_entries.push(EncodedBlockIndexEntry {
                extra_prefix: extra_prefix.clone(),
                min_suffix: min_suffix.clone(),
                max_suffix: max_suffix.clone(),
                byte_range: *block_offset..block_end,
            });
        }
        if entries[0].3 != SEGMENT_HEADER_LEN as u64 {
            return Err(CorruptionError::SegmentFormat);
        }
        let backing = bytes.into();
        let index = SegmentIndex::from_encoded_parts(backing, segment_prefix, encoded_entries)
            .ok_or(CorruptionError::SegmentFormat)?;
        if !index.validate_ranges(self.key_len, expected_min_key, expected_max_key) {
            return Err(CorruptionError::SegmentFormat);
        }
        Ok(index)
    }

    fn read_range(
        cursor: &mut BinaryCursor<'_>,
        len: usize,
    ) -> Result<std::ops::Range<usize>, CorruptionError> {
        let start = cursor.position();
        cursor
            .read_slice(len)
            .ok_or(CorruptionError::SegmentFormat)?;
        Ok(start..cursor.position())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        block::{BlockChecksumKind, ValuePayloadCompressionKind},
        segment::index::SegmentIndexBuilder,
    };

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
            let mut builder = SegmentIndexBuilder::new(b"a0", b"b9").expect("valid range");
            builder
                .push(b"a0", b"a9", SEGMENT_HEADER_LEN as u64..40)
                .expect("valid first block");
            builder
                .push(b"b0", b"b9", 40..64)
                .expect("valid second block");
            let footer = SegmentFooter {
                block_index: builder.finish(),
            };
            let mut bytes = Vec::new();
            footer
                .write_to(2, &mut bytes)
                .expect("footer should encode");
            assert_eq!(bytes.len(), 4 + 4 + 2 * (4 + 1 + 1 + 1 + 8) + 8);

            let decoded = SegmentFooter::from_bytes(bytes, 2, 64, b"a0", b"b9")
                .expect("footer offsets should define valid ranges");
            assert_eq!(
                decoded.block_index.entry(0).byte_range,
                SEGMENT_HEADER_LEN as u64..40
            );
            assert_eq!(decoded.block_index.entry(1).byte_range, 40..64);
        }

        #[test]
        fn rejects_impossible_block_count_before_allocating_entries() {
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&0u32.to_le_bytes());
            bytes.extend_from_slice(&u32::MAX.to_le_bytes());
            let footer_body_len = u32::try_from(bytes.len()).expect("small test footer");
            bytes.extend_from_slice(&footer_body_len.to_le_bytes());
            let crc = crc32c(&bytes);
            bytes.extend_from_slice(&crc.to_le_bytes());

            assert!(matches!(
                SegmentFooter::from_bytes(bytes, 2, SEGMENT_HEADER_LEN as u64, b"a0", b"a0"),
                Err(CorruptionError::SegmentFormat)
            ));
        }
    }
}
