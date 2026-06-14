//! Block-internal byte-layout math shared by the encoder and decoder.

use std::ops::Range;

use crc32c::crc32c_append;

use crate::format::{
    CorruptionError, FormatError, ValueLayout, binary::BinaryCursor, format_u32,
    record::EntrySource,
};

pub(crate) const KEY_PREFIX_LEN_LEN: usize = 4;
pub(crate) const CHECKSUM_LEN: usize = 4;

/// Derived lookup-metadata layout for one block.
///
/// The persisted block stores the key layout descriptor at the beginning of
/// the key section and the lookup checksum immediately after lookup metadata:
///
/// ```text
/// key_prefix_len:u32
/// key_prefix
/// key_suffixes
/// optional value_offsets
/// lookup_crc32c
/// ```
#[derive(Clone, Copy, Debug)]
pub(crate) struct BlockLookupLayout {
    pub(crate) key_prefix_len: usize,
    pub(crate) key_section_len: usize,
    pub(crate) lookup_metadata_len: usize,
}

impl BlockLookupLayout {
    pub(crate) fn new(
        record_count: usize,
        key_len: usize,
        value_layout: ValueLayout,
        key_prefix_len: usize,
    ) -> Result<Self, CorruptionError> {
        if key_prefix_len > key_len {
            return Err(CorruptionError::Block);
        }
        let suffix_len = key_len - key_prefix_len;
        let suffix_table_len = record_count
            .checked_mul(suffix_len)
            .ok_or(CorruptionError::Block)?;
        let key_section_len = KEY_PREFIX_LEN_LEN
            .checked_add(key_prefix_len)
            .and_then(|len| len.checked_add(suffix_table_len))
            .ok_or(CorruptionError::Block)?;
        let value_index_len = value_layout
            .offset_count(record_count)
            .and_then(|count| count.checked_mul(4))
            .ok_or(CorruptionError::Block)?;
        let lookup_metadata_len = key_section_len
            .checked_add(value_index_len)
            .ok_or(CorruptionError::Block)?;
        Ok(Self {
            key_prefix_len,
            key_section_len,
            lookup_metadata_len,
        })
    }

    pub(crate) fn read_key_prefix_len(bytes: &[u8]) -> Result<usize, CorruptionError> {
        let mut cursor = BinaryCursor::new(bytes);
        Ok(cursor.read::<u32>().ok_or(CorruptionError::Block)? as usize)
    }

    pub(crate) fn metadata_with_crc_len(self) -> Result<usize, CorruptionError> {
        self.lookup_metadata_len
            .checked_add(CHECKSUM_LEN)
            .ok_or(CorruptionError::Block)
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
        value_index_offset: usize,
        payload_offset: usize,
        payload_len: usize,
    ) -> Option<Self> {
        let payload = ValuePayload {
            offset: payload_offset,
            len: payload_len,
        };
        if let Some(value_len) = layout.fixed_width() {
            if record_count.checked_mul(value_len)? != payload_len {
                return None;
            }
            return Some(Self::Fixed {
                payload,
                value_len,
                record_count,
            });
        }
        Some(Self::Variable {
            index: ValueIndex {
                offset: value_index_offset,
                count: layout.offset_count(record_count)?,
            },
            payload,
        })
    }

    pub(super) fn from_metadata(
        layout: ValueLayout,
        record_count: usize,
        value_index_offset: usize,
        payload_offset: usize,
        metadata: &[u8],
    ) -> Option<Self> {
        let payload_len = if let Some(width) = layout.fixed_width() {
            record_count.checked_mul(width)?
        } else {
            let index = ValueIndex {
                offset: value_index_offset,
                count: layout.offset_count(record_count)?,
            };
            index.read(metadata, record_count).ok()?
        };
        Self::for_write(
            layout,
            record_count,
            value_index_offset,
            payload_offset,
            payload_len,
        )
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

pub(crate) fn read_stored_crc(bytes: &[u8], offset: usize) -> Result<u32, CorruptionError> {
    let end = offset
        .checked_add(CHECKSUM_LEN)
        .ok_or(CorruptionError::Block)?;
    let bytes = bytes.get(offset..end).ok_or(CorruptionError::Block)?;
    Ok(u32::from_le_bytes(bytes.try_into().expect("crc32c width")))
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
            let region = BlockValueRegion::for_write(ValueLayout::VARIABLE, 2, 16, 32, 5)
                .expect("layout should fit");
            let mut bytes = vec![0; 16];
            region
                .write_index(&entries, &mut bytes)
                .expect("index should encode");
            assert_eq!(bytes.len(), 28);
            bytes.resize(32, 0);
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
            let region = BlockValueRegion::from_metadata(value_layout, 2, 16, 16, &[])
                .expect("layout should decode");
            let mut bytes = vec![0; 16];
            bytes.extend_from_slice(b"aaaabbbb");

            assert_eq!(
                region.range(&bytes, 1).expect("second fixed value range"),
                4..8
            );
            assert!(
                BlockValueRegion::for_write(value_layout, 2, 16, 16, 7).is_none(),
                "fixed value payload must match record_count * value_len"
            );
        }
    }
}
