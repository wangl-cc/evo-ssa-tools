//! Block-internal byte-layout math shared by the encoder and decoder.

use std::ops::Range;

use crc32c::crc32c_append;

use crate::format::{
    CorruptionError, FormatError, ValueLayout, binary::BinaryCursor, format_u32,
    record::EntrySource,
};

pub(crate) const BLOCK_FOOTER_LEN: usize = 20;
const BLOCK_FOOTER_DESCRIPTOR_LEN: usize = 12;

/// Fixed-size block-local decoding metadata at the end of each block.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BlockFooter {
    pub(crate) key_prefix_len: usize,
    pub(crate) payload_offset: usize,
    pub(crate) payload_len: usize,
    metadata_crc32c: u32,
    payload_crc32c: u32,
}

impl BlockFooter {
    pub(super) fn new(
        key_prefix_len: usize,
        value_region: BlockValueRegion,
        block: &[u8],
    ) -> Result<Self, FormatError> {
        let footer = Self {
            key_prefix_len,
            payload_offset: value_region.payload_offset(),
            payload_len: value_region.payload_len(),
            metadata_crc32c: 0,
            payload_crc32c: 0,
        };
        let payload_end = footer
            .payload_offset
            .checked_add(footer.payload_len)
            .ok_or(FormatError::limit("block value payload end"))?;
        let footer = Self {
            metadata_crc32c: metadata_crc(
                block
                    .get(..footer.payload_offset)
                    .ok_or(FormatError::limit("block metadata checksum range"))?,
                &footer.descriptor_bytes()?,
            ),
            payload_crc32c: block_crc(
                block
                    .get(footer.payload_offset..payload_end)
                    .ok_or(FormatError::limit("block value checksum range"))?,
            ),
            ..footer
        };
        let block_len = block
            .len()
            .checked_add(BLOCK_FOOTER_LEN)
            .ok_or(FormatError::limit("block length"))?;
        footer.validate_len(block_len).map_err(|_| {
            FormatError::limit("block footer payload range should fit encoded block")
        })?;
        Ok(footer)
    }

    pub(super) fn from_bytes(bytes: &[u8], verify_checksum: bool) -> Result<Self, CorruptionError> {
        if bytes.len() < BLOCK_FOOTER_LEN {
            return Err(CorruptionError::Block);
        }
        let footer_offset = bytes.len() - BLOCK_FOOTER_LEN;
        let footer = Self::from_footer_bytes(&bytes[footer_offset..], bytes.len())?;
        if verify_checksum {
            footer.verify_metadata(
                bytes
                    .get(..footer.payload_offset)
                    .ok_or(CorruptionError::Block)?,
            )?;
            footer.verify_payload(
                bytes
                    .get(footer.payload_offset..footer.payload_end()?)
                    .ok_or(CorruptionError::Block)?,
            )?;
        }
        Ok(footer)
    }

    pub(crate) fn from_footer_bytes(
        bytes: &[u8],
        block_len: usize,
    ) -> Result<Self, CorruptionError> {
        if bytes.len() != BLOCK_FOOTER_LEN {
            return Err(CorruptionError::Block);
        }
        let mut cursor = BinaryCursor::new(bytes);
        let footer = Self {
            key_prefix_len: cursor.read::<u32>().ok_or(CorruptionError::Block)? as usize,
            payload_offset: cursor.read::<u32>().ok_or(CorruptionError::Block)? as usize,
            payload_len: cursor.read::<u32>().ok_or(CorruptionError::Block)? as usize,
            metadata_crc32c: cursor.read::<u32>().ok_or(CorruptionError::Block)?,
            payload_crc32c: cursor.read::<u32>().ok_or(CorruptionError::Block)?,
        };
        footer.validate_len(block_len)?;
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
        block.extend_from_slice(&self.metadata_crc32c.to_le_bytes());
        block.extend_from_slice(&self.payload_crc32c.to_le_bytes());
        Ok(())
    }

    pub(crate) fn verify_metadata(self, metadata: &[u8]) -> Result<(), CorruptionError> {
        if metadata.len() != self.payload_offset {
            return Err(CorruptionError::Block);
        }
        let descriptor = self
            .descriptor_bytes()
            .map_err(|_| CorruptionError::Block)?;
        if metadata_crc(metadata, &descriptor) != self.metadata_crc32c {
            return Err(CorruptionError::Block);
        }
        Ok(())
    }

    pub(crate) fn verify_payload(self, payload: &[u8]) -> Result<(), CorruptionError> {
        if payload.len() != self.payload_len || block_crc(payload) != self.payload_crc32c {
            return Err(CorruptionError::Block);
        }
        Ok(())
    }

    pub(crate) fn payload_end(self) -> Result<usize, CorruptionError> {
        self.payload_offset
            .checked_add(self.payload_len)
            .ok_or(CorruptionError::Block)
    }

    fn validate_len(self, block_len: usize) -> Result<(), CorruptionError> {
        let footer_offset = block_len
            .checked_sub(BLOCK_FOOTER_LEN)
            .ok_or(CorruptionError::Block)?;
        let Some(payload_end) = self.payload_offset.checked_add(self.payload_len) else {
            return Err(CorruptionError::Block);
        };
        if self.payload_offset > footer_offset || payload_end > footer_offset {
            return Err(CorruptionError::Block);
        }
        Ok(())
    }

    fn descriptor_bytes(self) -> Result<[u8; BLOCK_FOOTER_DESCRIPTOR_LEN], FormatError> {
        let mut bytes = [0u8; BLOCK_FOOTER_DESCRIPTOR_LEN];
        bytes[..4].copy_from_slice(
            &format_u32(self.key_prefix_len, "block footer key prefix length")?.to_le_bytes(),
        );
        bytes[4..8].copy_from_slice(
            &format_u32(self.payload_offset, "block footer value payload offset")?.to_le_bytes(),
        );
        bytes[8..12].copy_from_slice(
            &format_u32(self.payload_len, "block footer value payload length")?.to_le_bytes(),
        );
        Ok(bytes)
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

    pub(super) fn range(
        self,
        metadata: &[u8],
        index: usize,
    ) -> Result<Range<usize>, CorruptionError> {
        match self {
            Self::Variable {
                index: table,
                payload,
            } => {
                let start = table.read(metadata, index)?;
                let end = table.read(metadata, index + 1)?;
                payload.relative_range(start, end)
            }
            Self::Fixed {
                value_len,
                record_count,
                ..
            } => {
                if index >= record_count {
                    return Err(CorruptionError::Block);
                }
                let start = index.checked_mul(value_len).ok_or(CorruptionError::Block)?;
                self.payload().relative_range(start, start + value_len)
            }
        }
    }

    pub(super) fn validate(self, metadata: &[u8]) -> Result<(), CorruptionError> {
        match self {
            Self::Variable { index, payload } => index.validate(metadata, payload.len),
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
    fn relative_range(self, start: usize, end: usize) -> Result<Range<usize>, CorruptionError> {
        if start > end || end > self.len {
            return Err(CorruptionError::Block);
        }
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

fn metadata_crc(metadata: &[u8], descriptor: &[u8; BLOCK_FOOTER_DESCRIPTOR_LEN]) -> u32 {
    crc32c_append(crc32c_append(0, metadata), descriptor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::record::EntryRef;

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

            assert_eq!(region.range(&bytes, 0).expect("first value range"), 0..2);
            assert_eq!(region.range(&bytes, 1).expect("second value range"), 2..5);
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
                4..8
            );
            assert!(
                BlockValueRegion::from_footer(value_layout, 2, 16, 16, 7).is_none(),
                "fixed value payload must match record_count * value_len"
            );
        }
    }
}
