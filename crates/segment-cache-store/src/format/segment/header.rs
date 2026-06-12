use crc32c::crc32c;

use super::{HEADER_MAGIC, SEGMENT_FORMAT_VERSION, SEGMENT_HEADER_LEN};
use crate::format::{CorruptionError, FormatError, ValueLayout, binary::BinaryCursor};

/// Parsed fixed-size segment header.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SegmentHeader {
    version: u32,
    key_len: u32,
    value_layout: ValueLayout,
}

impl SegmentHeader {
    pub(crate) fn new(key_len: usize, value_layout: ValueLayout) -> Result<Self, FormatError> {
        Ok(Self {
            version: SEGMENT_FORMAT_VERSION,
            key_len: u32::try_from(key_len)
                .map_err(|_| FormatError::limit("segment key length"))?,
            value_layout,
        })
    }

    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self, CorruptionError> {
        if bytes.len() != SEGMENT_HEADER_LEN || &bytes[..4] != HEADER_MAGIC {
            return Err(CorruptionError::SegmentFormat);
        }
        let mut cursor = BinaryCursor::new(bytes);
        cursor.seek(20);
        let stored_crc = cursor.read::<u32>().ok_or(CorruptionError::SegmentFormat)?;
        if crc32c(&bytes[..20]) != stored_crc {
            return Err(CorruptionError::SegmentFormat);
        }
        cursor.seek(4);
        let version = cursor.read::<u32>().ok_or(CorruptionError::SegmentFormat)?;
        let key_len = cursor.read::<u32>().ok_or(CorruptionError::SegmentFormat)?;
        let value_len = cursor.read::<u32>().ok_or(CorruptionError::SegmentFormat)?;
        let reserved = cursor.read::<u32>().ok_or(CorruptionError::SegmentFormat)?;
        if reserved != 0 {
            return Err(CorruptionError::SegmentFormat);
        }
        Ok(Self {
            version,
            key_len,
            value_layout: ValueLayout::from_u32(value_len),
        })
    }

    pub(crate) fn into_bytes(self, buffer: &mut Vec<u8>) {
        let start = buffer.len();
        buffer.extend_from_slice(HEADER_MAGIC);
        buffer.extend_from_slice(&self.version.to_le_bytes());
        buffer.extend_from_slice(&self.key_len.to_le_bytes());
        buffer.extend_from_slice(&self.value_layout.to_u32().to_le_bytes());
        buffer.extend_from_slice(&0u32.to_le_bytes());
        let crc = crc32c(&buffer[start..start + 20]);
        buffer.extend_from_slice(&crc.to_le_bytes());
    }

    pub(crate) fn matches_geometry(
        self,
        expected_key_len: usize,
        expected_value_layout: ValueLayout,
    ) -> bool {
        self.version == SEGMENT_FORMAT_VERSION
            && self.key_len as usize == expected_key_len
            && self.value_layout == expected_value_layout
    }
}
