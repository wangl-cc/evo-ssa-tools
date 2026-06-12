use crc32c::crc32c;

use super::{SEGMENT_FOOTER_TRAILER_LEN, index::BlockIndexCodec};
use crate::format::{CorruptionError, FormatError, binary::BinaryCursor, format_u32};

/// Parsed segment footer.
///
/// The footer is the completion gate for a segment and owns the block index.
#[derive(Clone, Debug)]
pub(crate) struct SegmentFooter {
    pub(crate) record_count: u64,
    pub(crate) min_key: Vec<u8>,
    pub(crate) max_key: Vec<u8>,
    pub(crate) block_index: Vec<super::BlockIndexEntry>,
}

impl SegmentFooter {
    pub(crate) fn into_bytes(
        self,
        key_len: usize,
        buffer: &mut Vec<u8>,
    ) -> Result<(), FormatError> {
        let footer_body_start = buffer.len();
        buffer.extend_from_slice(&self.record_count.to_le_bytes());
        buffer.extend_from_slice(&self.min_key);
        buffer.extend_from_slice(&self.max_key);
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
        let min_key = cursor
            .read_vec(key_len)
            .ok_or(CorruptionError::SegmentFormat)?;
        let max_key = cursor
            .read_vec(key_len)
            .ok_or(CorruptionError::SegmentFormat)?;
        let block_index = BlockIndexCodec::new(key_len).read_entries(&mut cursor)?;
        let footer = Self {
            record_count,
            min_key,
            max_key,
            block_index,
        };
        BlockIndexCodec::new(key_len).validate_entries(&footer.block_index, &footer, data_end)?;
        Ok(footer)
    }
}
