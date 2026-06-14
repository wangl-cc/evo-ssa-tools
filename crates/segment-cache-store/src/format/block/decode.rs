//! Block decoder: on-disk block bytes in, zero-copy records out.
//!
//! Decoding validates the block footer, the value region, and key ordering
//! before any record is returned; a block that fails any check is reported as
//! [`CorruptionError::Block`] and degrades to a cache miss.

use std::{cell::OnceCell, cmp::Ordering};

use super::layout::{BlockFooter, BlockValueRegion};
use crate::format::{CorruptionError, ValueLayout, segment::BlockIndexEntry};

/// Decoded block bytes plus derived offsets needed for zero-copy record access.
#[derive(Debug)]
pub(crate) struct DecodedBlock {
    record_count: usize,
    key_len: usize,
    key_prefix_len: usize,
    suffix_len: usize,
    key_region_len: usize,
    value_region: BlockValueRegion,
    /// Materialized only when the block strips a shared key prefix and the last
    /// key is therefore not contiguous in the on-disk key region.
    last_key: Option<Vec<u8>>,
    /// Materialized lazily for scan-style APIs that must return borrowed full
    /// keys. Ordered lookup compares prefix/suffix bytes directly.
    full_keys: OnceCell<Vec<u8>>,
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
        layout.validate_key_region(&bytes)?;
        let last_key = layout.materialize_last_key_if_needed(&bytes, key_len)?;
        let block = Self {
            record_count,
            key_len,
            key_prefix_len: layout.key_prefix_len,
            suffix_len: layout.suffix_len,
            key_region_len: layout.key_region_len,
            value_region: layout.value_region,
            last_key,
            full_keys: OnceCell::new(),
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
        &self.bytes[..self.key_len]
    }

    pub(crate) fn last_key(&self) -> &[u8] {
        if let Some(last_key) = &self.last_key {
            return last_key;
        }
        let start = (self.record_count - 1) * self.key_len;
        &self.bytes[start..start + self.key_len]
    }

    pub(crate) fn find_value(&self, key: &[u8]) -> Option<Vec<u8>> {
        let index = self.partition_point_by_key(key);
        self.value_at_if_key(index, key).map(ToOwned::to_owned)
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

    pub(crate) fn compare_key_at_index(&self, index: usize, key: &[u8]) -> Ordering {
        self.compare_key(index, key).unwrap_or(Ordering::Greater)
    }

    pub(crate) fn value_at_if_key(&self, index: usize, key: &[u8]) -> Option<&[u8]> {
        if self.compare_key(index, key).ok()? != Ordering::Equal {
            return None;
        }
        let range = self.value_region.range(&self.bytes, index).ok()?;
        Some(&self.bytes[range])
    }

    fn partition_point_by_key(&self, key: &[u8]) -> usize {
        let mut left = 0usize;
        let mut right = self.record_count;
        while left < right {
            let mid = left + (right - left) / 2;
            match self.compare_key(mid, key) {
                Ok(Ordering::Less) => left = mid + 1,
                _ => right = mid,
            }
        }
        left
    }

    fn key_at(&self, index: usize) -> Result<&[u8], CorruptionError> {
        if index >= self.record_count {
            return Err(CorruptionError::Block);
        }
        if !self.needs_full_key_buffer() {
            let start = index * self.key_len;
            let end = start + self.key_len;
            if end > self.key_region_len {
                return Err(CorruptionError::Block);
            }
            return Ok(&self.bytes[start..end]);
        }

        let start = index * self.key_len;
        let end = start + self.key_len;
        let full_keys = self.full_keys.get_or_init(|| self.reconstruct_full_keys());
        if end > full_keys.len() {
            return Err(CorruptionError::Block);
        }
        Ok(&full_keys[start..end])
    }

    fn compare_key(&self, index: usize, key: &[u8]) -> Result<Ordering, CorruptionError> {
        if index >= self.record_count || key.len() != self.key_len {
            return Err(CorruptionError::Block);
        }
        let prefix = &self.bytes[..self.key_prefix_len];
        let prefix_order = prefix.cmp(&key[..self.key_prefix_len]);
        if prefix_order != Ordering::Equal {
            return Ok(prefix_order);
        }
        self.key_suffix_at(index)
            .map(|suffix| suffix.cmp(&key[self.key_prefix_len..]))
    }

    fn key_suffix_at(&self, index: usize) -> Result<&[u8], CorruptionError> {
        if index >= self.record_count {
            return Err(CorruptionError::Block);
        }
        let start = self
            .key_prefix_len
            .checked_add(
                index
                    .checked_mul(self.suffix_len)
                    .ok_or(CorruptionError::Block)?,
            )
            .ok_or(CorruptionError::Block)?;
        let end = start
            .checked_add(self.suffix_len)
            .ok_or(CorruptionError::Block)?;
        if end > self.key_region_len {
            return Err(CorruptionError::Block);
        }
        Ok(&self.bytes[start..end])
    }

    fn needs_full_key_buffer(&self) -> bool {
        self.key_prefix_len > 0 && self.record_count > 1
    }

    fn reconstruct_full_keys(&self) -> Vec<u8> {
        let prefix = &self.bytes[..self.key_prefix_len];
        let mut full_keys = Vec::with_capacity(self.record_count * self.key_len);
        for index in 0..self.record_count {
            let suffix_start = self.key_prefix_len + index * self.suffix_len;
            let suffix_end = suffix_start + self.suffix_len;
            full_keys.extend_from_slice(prefix);
            full_keys.extend_from_slice(&self.bytes[suffix_start..suffix_end]);
        }
        full_keys
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
        if self.compare_key(0, entry.first_key.as_slice())? != Ordering::Equal {
            return Err(CorruptionError::Block);
        }
        let mut previous = self.key_suffix_at(0)?;
        for index in 1..self.record_count {
            let current = self.key_suffix_at(index)?;
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

    fn validate_key_region(&self, bytes: &[u8]) -> Result<(), CorruptionError> {
        if self.key_region_len > bytes.len() {
            return Err(CorruptionError::Block);
        }
        Ok(())
    }

    fn materialize_last_key_if_needed(
        &self,
        bytes: &[u8],
        key_len: usize,
    ) -> Result<Option<Vec<u8>>, CorruptionError> {
        if self.key_prefix_len == 0 || self.record_count == 1 {
            return Ok(None);
        }
        let index = self.record_count - 1;
        let prefix = &bytes[..self.key_prefix_len];
        let suffix_start = self
            .key_prefix_len
            .checked_add(
                index
                    .checked_mul(self.suffix_len)
                    .ok_or(CorruptionError::Block)?,
            )
            .ok_or(CorruptionError::Block)?;
        let suffix_end = suffix_start
            .checked_add(self.suffix_len)
            .ok_or(CorruptionError::Block)?;
        if suffix_end > self.key_region_len {
            return Err(CorruptionError::Block);
        }
        let mut key = Vec::with_capacity(key_len);
        key.extend_from_slice(prefix);
        key.extend_from_slice(&bytes[suffix_start..suffix_end]);
        Ok(Some(key))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{
        block::BlockBuilder,
        record::{EntryRef, EntrySource},
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

            assert!(
                decoded.full_keys.get().is_none(),
                "lookup methods should not materialize full key table"
            );
            assert_eq!(decoded.first_key(), b"aa01");
            assert_eq!(decoded.last_key(), b"aa02");
            assert_eq!(decoded.find_value(b"aa02"), Some(b"second".to_vec()));
            assert!(
                decoded.full_keys.get().is_none(),
                "find_value should compare prefix-stripped keys without materialization"
            );
            assert_eq!(
                decoded
                    .record_at_index(1)
                    .expect("record should decode")
                    .key,
                b"aa02"
            );
            assert!(
                decoded.full_keys.get().is_some(),
                "scan-style record access materializes full keys lazily"
            );
        }

        #[test]
        fn single_record_block_borrows_raw_key_region() {
            let entries = Entries {
                entries: &[(b"aa01", b"only")],
            };
            let block = BlockBuilder::new(&entries, 4, ValueLayout::VARIABLE, 0)
                .encode()
                .expect("block should encode");
            let entry = BlockIndexEntry {
                first_key: b"aa01".to_vec(),
                block_offset: 0,
                block_len: u32::try_from(block.len()).expect("block len"),
                record_count: 1,
            };

            let decoded = DecodedBlock::decode(block, &entry, 4, ValueLayout::VARIABLE, true)
                .expect("block should decode");

            assert!(decoded.full_keys.get().is_none());
            assert_eq!(decoded.first_key(), b"aa01");
            assert_eq!(decoded.last_key(), b"aa01");
            assert_eq!(decoded.find_value(b"aa01"), Some(b"only".to_vec()));
        }
    }
}
