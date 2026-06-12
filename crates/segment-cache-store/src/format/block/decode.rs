//! Block decoder: on-disk block bytes in, zero-copy records out.
//!
//! Decoding validates the block footer, the value region, and full key
//! ordering before any record is returned; a block that fails any check is
//! reported as [`CorruptionError::Block`] and degrades to a cache miss.

use super::layout::{BlockFooter, BlockValueRegion};
use crate::format::{CorruptionError, ValueLayout, segment::BlockIndexEntry};

/// Decoded block bytes plus derived offsets needed for zero-copy record access.
#[derive(Debug)]
pub(crate) struct DecodedBlock {
    record_count: usize,
    key_len: usize,
    key_prefix_len: usize,
    suffix_len: usize,
    last_key_offset: usize,
    key_region_len: usize,
    value_region: BlockValueRegion,
    full_keys: Vec<u8>,
    bytes: Vec<u8>,
}

/// Borrowed key/value record decoded from stored bytes.
pub(crate) struct ParsedRecord<'a> {
    pub(crate) key: &'a [u8],
    pub(crate) value: &'a [u8],
}

impl DecodedBlock {
    pub(crate) fn decode(
        bytes: Vec<u8>,
        entry: &BlockIndexEntry,
        key_len: usize,
        value_layout: ValueLayout,
        verify_checksum: bool,
    ) -> Result<Self, CorruptionError> {
        let footer = BlockFooter::from_bytes(&bytes, verify_checksum)?;
        let record_count = entry.record_count as usize;
        if record_count == 0 {
            return Err(CorruptionError::Block);
        }
        let layout = BlockLayout::new(record_count, key_len, value_layout, footer)?;
        let full_keys = layout.reconstruct_full_keys(&bytes)?;
        let block = Self {
            record_count,
            key_len,
            key_prefix_len: layout.key_prefix_len,
            suffix_len: layout.suffix_len,
            last_key_offset: (record_count - 1) * key_len,
            key_region_len: layout.key_region_len,
            value_region: layout.value_region,
            full_keys,
            bytes,
        };
        block.value_region.validate(&block.bytes)?;
        block.validate_key_ordering(entry)?;
        Ok(block)
    }

    pub(crate) fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    pub(crate) fn first_key(&self) -> &[u8] {
        debug_assert!(self.record_count > 0);
        if !self.full_keys.is_empty() {
            return &self.full_keys[..self.key_len];
        }
        &self.bytes[..self.key_len]
    }

    pub(crate) fn last_key(&self) -> &[u8] {
        let start = self.last_key_offset;
        if self.full_keys.is_empty() {
            let raw_start = self.key_prefix_len + (self.record_count - 1) * self.suffix_len;
            return &self.bytes[raw_start..raw_start + self.key_len];
        }
        &self.full_keys[start..start + self.key_len]
    }

    pub(crate) fn find_value(&self, key: &[u8]) -> Option<Vec<u8>> {
        let index = self.partition_point_by_key(key);
        let record = self.record_at(index).ok()?;
        if record.key == key {
            Some(record.value.to_vec())
        } else {
            None
        }
    }

    pub(crate) fn records_from(
        &self,
        start_index: usize,
    ) -> impl Iterator<Item = ParsedRecord<'_>> {
        (start_index..self.record_count).filter_map(|index| self.record_at(index).ok())
    }

    pub(crate) fn record_count(&self) -> usize {
        self.record_count
    }

    pub(crate) fn key_at_index(&self, index: usize) -> Result<&[u8], CorruptionError> {
        self.key_at(index)
    }

    pub(crate) fn record_at_index(
        &self,
        index: usize,
    ) -> Result<ParsedRecord<'_>, CorruptionError> {
        self.record_at(index)
    }

    pub(crate) fn lower_bound_index(&self, key: &[u8]) -> usize {
        self.partition_point_by_key(key)
    }

    fn partition_point_by_key(&self, key: &[u8]) -> usize {
        let mut left = 0usize;
        let mut right = self.record_count;
        while left < right {
            let mid = left + (right - left) / 2;
            match self.key_at(mid) {
                Ok(candidate) if candidate < key => left = mid + 1,
                _ => right = mid,
            }
        }
        left
    }

    fn key_at(&self, index: usize) -> Result<&[u8], CorruptionError> {
        if index >= self.record_count {
            return Err(CorruptionError::Block);
        }
        let start = index * self.key_len;
        let end = start + self.key_len;
        if !self.full_keys.is_empty() {
            if end > self.full_keys.len() {
                return Err(CorruptionError::Block);
            }
            return Ok(&self.full_keys[start..end]);
        }

        if end > self.key_region_len {
            return Err(CorruptionError::Block);
        }
        Ok(&self.bytes[start..end])
    }

    fn record_at(&self, index: usize) -> Result<ParsedRecord<'_>, CorruptionError> {
        let key = self.key_at(index)?;
        let range = self.value_region.range(&self.bytes, index)?;
        Ok(ParsedRecord {
            key,
            value: &self.bytes[range],
        })
    }

    fn validate_key_ordering(&self, entry: &BlockIndexEntry) -> Result<(), CorruptionError> {
        let first_key = self.key_at(0)?;
        if first_key != entry.first_key.as_slice() {
            return Err(CorruptionError::Block);
        }
        let mut previous = first_key;
        for index in 1..self.record_count {
            let current = self.key_at(index)?;
            if current <= previous {
                return Err(CorruptionError::Block);
            }
            previous = current;
        }
        Ok(())
    }
}

/// Block layout derived from the block footer during decoding.
struct BlockLayout {
    record_count: usize,
    key_prefix_len: usize,
    suffix_len: usize,
    key_region_len: usize,
    value_region: BlockValueRegion,
}

impl BlockLayout {
    fn new(
        record_count: usize,
        key_len: usize,
        value_layout: ValueLayout,
        footer: BlockFooter,
    ) -> Result<Self, CorruptionError> {
        if footer.key_prefix_len > key_len {
            return Err(CorruptionError::Block);
        }
        let suffix_len = key_len - footer.key_prefix_len;
        let suffix_table_len = record_count
            .checked_mul(suffix_len)
            .ok_or(CorruptionError::Block)?;
        let key_region_len = footer
            .key_prefix_len
            .checked_add(suffix_table_len)
            .ok_or(CorruptionError::Block)?;
        let value_region = BlockValueRegion::from_footer(
            value_layout,
            record_count,
            key_region_len,
            footer.payload_offset,
            footer.payload_len,
        )
        .ok_or(CorruptionError::Block)?;
        Ok(Self {
            record_count,
            key_prefix_len: footer.key_prefix_len,
            suffix_len,
            key_region_len,
            value_region,
        })
    }

    fn reconstruct_full_keys(&self, bytes: &[u8]) -> Result<Vec<u8>, CorruptionError> {
        if self.key_prefix_len == 0 {
            return Ok(Vec::new());
        }
        if self.key_region_len > bytes.len() {
            return Err(CorruptionError::Block);
        }
        let prefix = &bytes[..self.key_prefix_len];
        let mut full_keys =
            Vec::with_capacity(self.record_count * (self.key_prefix_len + self.suffix_len));
        for index in 0..self.record_count {
            let suffix_start = self.key_prefix_len + index * self.suffix_len;
            let suffix_end = suffix_start + self.suffix_len;
            if suffix_end > self.key_region_len {
                return Err(CorruptionError::Block);
            }
            full_keys.extend_from_slice(prefix);
            full_keys.extend_from_slice(&bytes[suffix_start..suffix_end]);
        }
        Ok(full_keys)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{
        block::BlockBuilder,
        entry::{EntryRef, EntrySource},
    };

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

    mod prefix_stripping {
        use super::*;

        #[test]
        fn stores_common_prefix_once_and_round_trips_full_keys() {
            let entries = Entries {
                entries: &[(b"aa01", b"first"), (b"aa02", b"second")],
            };
            let block = BlockBuilder::new(&entries, 4, ValueLayout::VARIABLE, 0)
                .encode()
                .expect("block should encode");
            let entry = BlockIndexEntry {
                first_key: b"aa01".to_vec(),
                block_offset: 0,
                block_len: u32::try_from(block.len()).expect("block len"),
                record_count: 2,
            };

            assert_eq!(&block[..3], b"aa0");
            assert_eq!(&block[3..5], b"12");

            let decoded = DecodedBlock::decode(block, &entry, 4, ValueLayout::VARIABLE, true)
                .expect("block should decode");

            assert_eq!(decoded.first_key(), b"aa01");
            assert_eq!(decoded.last_key(), b"aa02");
            assert_eq!(decoded.find_value(b"aa02"), Some(b"second".to_vec()));
        }
    }
}
