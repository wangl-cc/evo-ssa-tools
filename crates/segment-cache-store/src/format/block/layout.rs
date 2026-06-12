//! Block-internal byte-layout math shared by the encoder and decoder.

use std::ops::Range;

use crc32c::crc32c_append;

use crate::format::{
    CorruptionError, FormatError, ValueLayout, binary::BinaryCursor, entry::EntrySource, format_u32,
};

pub(crate) const BLOCK_FOOTER_LEN: usize = 16;
const BLOCK_CHECKSUM_LEN: usize = 4;

/// Fixed-size block-local decoding metadata at the end of each block.
#[derive(Clone, Copy)]
pub(super) struct BlockFooter {
    pub(super) key_prefix_len: usize,
    pub(super) payload_offset: usize,
    pub(super) payload_len: usize,
}

impl BlockFooter {
    pub(super) fn new(key_prefix_len: usize, value_region: BlockValueRegion) -> Self {
        Self {
            key_prefix_len,
            payload_offset: value_region.payload_offset(),
            payload_len: value_region.payload_len(),
        }
    }

    pub(super) fn from_bytes(bytes: &[u8], verify_checksum: bool) -> Result<Self, CorruptionError> {
        if bytes.len() < BLOCK_FOOTER_LEN {
            return Err(CorruptionError::Block);
        }
        let checksum_offset = bytes.len() - BLOCK_CHECKSUM_LEN;
        let footer_offset = bytes.len() - BLOCK_FOOTER_LEN;
        let mut cursor = BinaryCursor::new(bytes);
        cursor.seek(checksum_offset);
        let stored_crc = cursor.read::<u32>().ok_or(CorruptionError::Block)?;
        if verify_checksum && block_crc(&bytes[..checksum_offset]) != stored_crc {
            return Err(CorruptionError::Block);
        }
        cursor.seek(footer_offset);
        let footer = Self {
            key_prefix_len: cursor.read::<u32>().ok_or(CorruptionError::Block)? as usize,
            payload_offset: cursor.read::<u32>().ok_or(CorruptionError::Block)? as usize,
            payload_len: cursor.read::<u32>().ok_or(CorruptionError::Block)? as usize,
        };
        footer.validate(bytes)?;
        Ok(footer)
    }

    pub(super) fn into_bytes(self, block: &mut Vec<u8>) -> Result<(), FormatError> {
        block.extend_from_slice(
            &format_u32(self.key_prefix_len, "block footer key prefix length")?.to_le_bytes(),
        );
        block.extend_from_slice(
            &format_u32(self.payload_offset, "block footer value payload offset")?.to_le_bytes(),
        );
        block.extend_from_slice(
            &format_u32(self.payload_len, "block footer value payload length")?.to_le_bytes(),
        );
        let checksum = block_crc(block);
        block.extend_from_slice(&checksum.to_le_bytes());
        Ok(())
    }

    fn validate(self, bytes: &[u8]) -> Result<(), CorruptionError> {
        let footer_offset = bytes.len() - BLOCK_FOOTER_LEN;
        let Some(payload_end) = self.payload_offset.checked_add(self.payload_len) else {
            return Err(CorruptionError::Block);
        };
        if self.payload_offset > footer_offset || payload_end > footer_offset {
            return Err(CorruptionError::Block);
        }
        Ok(())
    }
}

/// In-memory descriptor for locating values inside one block.
#[derive(Clone, Copy, Debug)]
pub(super) enum BlockValueRegion {
    /// Variable values use an on-disk value index and a packed value payload.
    Variable {
        index: ValueIndex,
        payload: ValuePayload,
    },
    /// Fixed values use arithmetic over the packed value payload.
    Fixed {
        payload: ValuePayload,
        value_len: usize,
        record_count: usize,
    },
}

/// Byte range of the packed value payload inside a block.
#[derive(Clone, Copy, Debug)]
pub(super) struct ValuePayload {
    offset: usize,
    len: usize,
}

/// Byte range of the variable-value index inside a block.
#[derive(Clone, Copy, Debug)]
pub(super) struct ValueIndex {
    offset: usize,
    count: usize,
}

impl BlockValueRegion {
    pub(super) fn for_write(
        layout: ValueLayout,
        record_count: usize,
        value_region_offset: usize,
        payload_len: usize,
    ) -> Option<Self> {
        let payload_offset = layout.value_payload_offset(record_count, value_region_offset)?;
        let payload = ValuePayload {
            offset: payload_offset,
            len: payload_len,
        };
        if let Some(value_len) = layout.fixed_width() {
            return Some(Self::Fixed {
                payload,
                value_len,
                record_count,
            });
        }
        Some(Self::Variable {
            index: ValueIndex {
                offset: value_region_offset,
                count: layout.offset_count(record_count)?,
            },
            payload,
        })
    }

    pub(super) fn from_footer(
        layout: ValueLayout,
        record_count: usize,
        value_region_offset: usize,
        payload_offset: usize,
        payload_len: usize,
    ) -> Option<Self> {
        if layout.value_payload_offset(record_count, value_region_offset)? != payload_offset {
            return None;
        }
        if let Some(width) = layout.fixed_width() {
            let expected_len = record_count.checked_mul(width)?;
            if payload_len != expected_len {
                return None;
            }
        }
        Self::for_write(layout, record_count, value_region_offset, payload_len)
    }

    pub(super) fn payload_offset(self) -> usize {
        self.payload().offset
    }

    pub(super) fn payload_len(self) -> usize {
        self.payload().len
    }

    pub(super) fn write_index<S: EntrySource + ?Sized>(
        self,
        entries: &S,
        block: &mut Vec<u8>,
    ) -> Result<(), FormatError> {
        let Self::Variable { .. } = self else {
            return Ok(());
        };
        let mut offset = 0usize;
        for index in 0..entries.len() {
            let value = entries.entry(index).value();
            block.extend_from_slice(&format_u32(offset, "value offset")?.to_le_bytes());
            offset = offset
                .checked_add(value.len())
                .ok_or(FormatError::limit("accumulated value offset"))?;
        }
        block.extend_from_slice(&format_u32(offset, "value payload length")?.to_le_bytes());
        Ok(())
    }

    pub(super) fn range(self, bytes: &[u8], index: usize) -> Result<Range<usize>, CorruptionError> {
        match self {
            Self::Variable {
                index: table,
                payload,
            } => {
                let start = table.read(bytes, index)?;
                let end = table.read(bytes, index + 1)?;
                payload.absolute_range(start, end)
            }
            Self::Fixed {
                payload,
                value_len,
                record_count,
            } => {
                if index >= record_count {
                    return Err(CorruptionError::Block);
                }
                let start = index.checked_mul(value_len).ok_or(CorruptionError::Block)?;
                payload.absolute_range(start, start + value_len)
            }
        }
    }

    pub(super) fn validate(self, bytes: &[u8]) -> Result<(), CorruptionError> {
        match self {
            Self::Variable { index, payload } => index.validate(bytes, payload.len),
            Self::Fixed {
                payload,
                value_len,
                record_count,
            } => {
                let expected_len = record_count
                    .checked_mul(value_len)
                    .ok_or(CorruptionError::Block)?;
                if payload.len != expected_len {
                    return Err(CorruptionError::Block);
                }
                Ok(())
            }
        }
    }

    fn payload(self) -> ValuePayload {
        match self {
            Self::Variable { payload, .. } | Self::Fixed { payload, .. } => payload,
        }
    }
}

impl ValuePayload {
    fn absolute_range(self, start: usize, end: usize) -> Result<Range<usize>, CorruptionError> {
        if start > end || end > self.len {
            return Err(CorruptionError::Block);
        }
        let start = self
            .offset
            .checked_add(start)
            .ok_or(CorruptionError::Block)?;
        let end = self.offset.checked_add(end).ok_or(CorruptionError::Block)?;
        Ok(start..end)
    }
}

impl ValueIndex {
    fn read(self, bytes: &[u8], index: usize) -> Result<usize, CorruptionError> {
        if index >= self.count {
            return Err(CorruptionError::Block);
        }
        let start = self
            .offset
            .checked_add(index.checked_mul(4).ok_or(CorruptionError::Block)?)
            .ok_or(CorruptionError::Block)?;
        let mut cursor = BinaryCursor::at(bytes, start);
        Ok(cursor.read::<u32>().ok_or(CorruptionError::Block)? as usize)
    }

    fn validate(self, bytes: &[u8], payload_len: usize) -> Result<(), CorruptionError> {
        let mut previous = 0usize;
        for index in 0..self.count {
            let offset = self.read(bytes, index)?;
            if offset < previous || offset > payload_len {
                return Err(CorruptionError::Block);
            }
            previous = offset;
        }
        if previous != payload_len {
            return Err(CorruptionError::Block);
        }
        Ok(())
    }
}

pub(super) fn block_crc(bytes: &[u8]) -> u32 {
    crc32c_append(0, bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::entry::EntryRef;

    struct Entries<'a> {
        entries: &'a [(&'a [u8], &'a [u8])],
    }

    impl EntrySource for Entries<'_> {
        fn len(&self) -> usize {
            self.entries.len()
        }

        fn entry(&self, index: usize) -> EntryRef<'_> {
            let (key, value) = self.entries[index];
            EntryRef::new(key, value)
        }
    }

    mod value_region {
        use super::*;

        #[test]
        fn variable_values_store_offsets_with_payload_end_sentinel() {
            let entries = Entries {
                entries: &[(b"k1", b"aa"), (b"k2", b"bbb")],
            };
            let region = BlockValueRegion::for_write(ValueLayout::VARIABLE, 2, 16, 5)
                .expect("layout should fit");
            let mut bytes = vec![0; 16];
            region
                .write_index(&entries, &mut bytes)
                .expect("index should encode");
            bytes.extend_from_slice(b"aabbb");

            assert_eq!(region.range(&bytes, 0).expect("first value range"), 28..30);
            assert_eq!(region.range(&bytes, 1).expect("second value range"), 30..33);
            region
                .validate(&bytes)
                .expect("variable value index should be valid");
        }

        #[test]
        fn fixed_values_compute_ranges_without_tables() {
            let value_layout =
                ValueLayout::fixed(std::num::NonZeroU32::new(4).expect("fixed len is non-zero"));
            let region = BlockValueRegion::from_footer(value_layout, 2, 16, 16, 8)
                .expect("layout should decode");
            let mut bytes = vec![0; 16];
            bytes.extend_from_slice(b"aaaabbbb");

            assert_eq!(
                region.range(&bytes, 1).expect("second fixed value range"),
                20..24
            );
            assert!(
                BlockValueRegion::from_footer(value_layout, 2, 16, 16, 7).is_none(),
                "fixed value payload must match record_count * value_len"
            );
        }
    }
}
