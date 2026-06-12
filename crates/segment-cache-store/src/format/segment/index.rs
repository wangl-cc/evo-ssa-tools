use super::{SEGMENT_HEADER_LEN, footer::SegmentFooter};
use crate::format::{
    CorruptionError, FormatError, binary::BinaryCursor, block::BLOCK_FOOTER_LEN, format_u32,
};

/// Sparse index entry for one data block in a segment file.
#[derive(Clone, Debug)]
pub(crate) struct BlockIndexEntry {
    pub(crate) first_key: Vec<u8>,
    pub(crate) block_offset: u64,
    pub(crate) block_len: u32,
    pub(crate) record_count: u32,
}

pub(super) struct BlockIndexCodec {
    key_len: usize,
}

impl BlockIndexCodec {
    pub(super) fn new(key_len: usize) -> Self {
        Self { key_len }
    }

    pub(super) fn write_to(
        &self,
        entries: &[BlockIndexEntry],
        buffer: &mut Vec<u8>,
    ) -> Result<(), FormatError> {
        buffer.extend_from_slice(
            &format_u32(entries.len(), "block index entry count")?.to_le_bytes(),
        );
        for entry in entries {
            buffer.extend_from_slice(&entry.first_key);
            buffer.extend_from_slice(&entry.block_offset.to_le_bytes());
            buffer.extend_from_slice(&entry.block_len.to_le_bytes());
            buffer.extend_from_slice(&entry.record_count.to_le_bytes());
        }
        Ok(())
    }

    pub(super) fn read_entries(
        &self,
        cursor: &mut BinaryCursor<'_>,
    ) -> Result<Vec<BlockIndexEntry>, CorruptionError> {
        let count = cursor.read::<u32>().ok_or(CorruptionError::SegmentFormat)?;
        let count = count as usize;
        let expected_remaining = count
            .checked_mul(self.key_len + 8 + 4 + 4)
            .ok_or(CorruptionError::SegmentFormat)?;
        if cursor.remaining() != expected_remaining {
            return Err(CorruptionError::SegmentFormat);
        }
        let mut entries = Vec::with_capacity(count);
        for _ in 0..count {
            let first_key = cursor
                .read_vec(self.key_len)
                .ok_or(CorruptionError::SegmentFormat)?;
            let block_offset = cursor.read::<u64>().ok_or(CorruptionError::SegmentFormat)?;
            let block_len = cursor.read::<u32>().ok_or(CorruptionError::SegmentFormat)?;
            let record_count = cursor.read::<u32>().ok_or(CorruptionError::SegmentFormat)?;
            entries.push(BlockIndexEntry {
                first_key,
                block_offset,
                block_len,
                record_count,
            });
        }
        Ok(entries)
    }

    pub(super) fn validate_entries(
        &self,
        entries: &[BlockIndexEntry],
        footer: &SegmentFooter,
        data_end: u64,
    ) -> Result<(), CorruptionError> {
        if entries.is_empty() || footer.record_count == 0 {
            return Err(CorruptionError::SegmentFormat);
        }
        if entries[0].first_key != footer.min_key {
            return Err(CorruptionError::SegmentFormat);
        }

        let mut expected_offset = SEGMENT_HEADER_LEN as u64;
        let mut decoded_records = 0u64;
        let mut previous_first_key: Option<&[u8]> = None;
        for entry in entries {
            if entry.block_len < BLOCK_FOOTER_LEN as u32
                || entry.record_count == 0
                || entry.block_offset != expected_offset
                || entry.first_key.len() != self.key_len
                || entry.first_key.as_slice() > footer.max_key.as_slice()
            {
                return Err(CorruptionError::SegmentFormat);
            }
            if let Some(previous_first_key) = previous_first_key
                && entry.first_key.as_slice() <= previous_first_key
            {
                return Err(CorruptionError::SegmentFormat);
            }
            expected_offset = expected_offset
                .checked_add(u64::from(entry.block_len))
                .ok_or(CorruptionError::SegmentFormat)?;
            decoded_records = decoded_records
                .checked_add(u64::from(entry.record_count))
                .ok_or(CorruptionError::SegmentFormat)?;
            previous_first_key = Some(entry.first_key.as_slice());
        }
        if expected_offset != data_end || decoded_records != footer.record_count {
            return Err(CorruptionError::SegmentFormat);
        }
        Ok(())
    }
}
