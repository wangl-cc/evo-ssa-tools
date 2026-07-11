//! Block decoder: on-disk block bytes in, borrowed record views out.
//!
//! Decoding validates the block lookup metadata, value layout, and key
//! ordering before any record is returned. A block that fails validation is
//! reported as [`CorruptionError::Block`] and degrades to a cache miss.

use std::{cell::OnceCell, cmp::Ordering};

use super::layout::{
    BlockLookupLayout, BlockValueRegion, KEY_PREFIX_LEN_LEN, read_stored_checksum,
};
use crate::format::{
    BlockChecksumKind, CorruptionError, DecodedPayload, ValueLayout, ValuePayloadCompressionKind,
    ValuePayloadDecoder, segment::BlockIndexEntry,
};

/// Raw block bytes plus derived offsets needed for borrowed record access.
///
/// Prefix-compressed keys are materialized lazily for scan APIs, while
/// compressed values borrow from a decoder-owned buffer supplied by the read
/// session.
#[derive(Debug)]
pub(crate) struct DecodedBlock {
    bytes: Vec<u8>,
    record_count: usize,
    key_len: usize,
    key_prefix_len: usize,
    suffix_len: usize,
    key_section_len: usize,
    payload_offset: usize,
    payload_len: usize,
    payload_data_offset: usize,
    value_checksum_offset: usize,
    payload_is_borrowable: bool,
    value_payload_compression: ValuePayloadCompressionKind,
    value_region: BlockValueRegion,
    /// Materialized only when the block strips a shared key prefix and the last
    /// key is therefore not contiguous in the on-disk key section.
    last_key: Option<Vec<u8>>,
    /// Materialized lazily for scan-style APIs that must return borrowed full
    /// keys. Ordered lookup compares prefix/suffix bytes directly.
    full_keys: OnceCell<Vec<u8>>,
}

/// Borrowed key/value record decoded from stored bytes.
pub(crate) struct ParsedRecord<'key, 'value> {
    pub(crate) key: &'key [u8],
    pub(crate) value: &'value [u8],
}

#[derive(Clone, Copy)]
pub(crate) enum BlockKeyUpperBound<'a> {
    /// Non-final blocks must end before the next block's first key.
    Exclusive(&'a [u8]),
    /// The final block may end at the segment max key.
    Inclusive(&'a [u8]),
}

#[derive(Clone, Copy)]
pub(crate) struct BlockDecodeOptions<'a> {
    pub(crate) key_len: usize,
    pub(crate) value_layout: ValueLayout,
    pub(crate) block_checksum: BlockChecksumKind,
    pub(crate) value_payload_compression: ValuePayloadCompressionKind,
    pub(crate) verify_checksum: bool,
    pub(crate) upper_key_bound: BlockKeyUpperBound<'a>,
}

impl DecodedBlock {
    pub(crate) fn decode(
        bytes: Vec<u8>,
        entry: &BlockIndexEntry,
        options: BlockDecodeOptions<'_>,
    ) -> Result<Self, CorruptionError> {
        let layout = BlockLayout::from_block_bytes(
            &bytes,
            entry,
            options.key_len,
            options.value_layout,
            options.block_checksum,
            options.value_payload_compression,
        )?;
        if options.verify_checksum {
            layout.verify_lookup_checksum(&bytes)?;
            layout.verify_payload_checksum(&bytes)?;
        }
        Self::from_bytes(bytes, layout, entry, options.upper_key_bound)
    }

    fn from_bytes(
        bytes: Vec<u8>,
        layout: BlockLayout,
        entry: &BlockIndexEntry,
        upper_key_bound: BlockKeyUpperBound<'_>,
    ) -> Result<Self, CorruptionError> {
        layout.validate_key_section(&bytes)?;
        let last_key = layout.materialize_last_key_if_needed(&bytes)?;
        let block = Self {
            bytes,
            record_count: layout.record_count,
            key_len: layout.key_len,
            key_prefix_len: layout.lookup.key_prefix_len,
            suffix_len: layout.suffix_len,
            key_section_len: layout.lookup.key_section_len,
            payload_offset: layout.payload_offset,
            payload_len: layout.payload_len,
            payload_data_offset: layout.payload_data_offset,
            value_checksum_offset: layout.value_checksum_offset,
            payload_is_borrowable: layout.payload_is_borrowable,
            value_payload_compression: layout.value_payload_compression,
            value_region: layout.value_region,
            last_key,
            full_keys: OnceCell::new(),
        };
        block.value_region.validate(block.metadata())?;
        block.validate_key_ordering(entry, upper_key_bound)?;
        Ok(block)
    }

    pub(crate) fn decode_payload_if_needed(
        &self,
        payload_decoder: &mut ValuePayloadDecoder,
    ) -> Result<Option<Vec<u8>>, CorruptionError> {
        if self.payload_is_borrowable {
            return Ok(None);
        }
        match self.value_payload_compression.decode_frame(
            payload_decoder,
            self.payload_frame()?,
            self.payload_len,
        )? {
            DecodedPayload::Borrowed(bytes) => {
                if bytes.len() != self.payload_len {
                    return Err(CorruptionError::Block);
                }
                Ok(None)
            }
            #[cfg(any(feature = "value-compression-lz4", feature = "value-compression-zstd"))]
            DecodedPayload::Owned(bytes) => Ok(Some(bytes)),
        }
    }

    pub(crate) fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    pub(crate) fn payload_frame(&self) -> Result<&[u8], CorruptionError> {
        self.bytes
            .get(self.payload_offset..self.value_checksum_offset)
            .ok_or(CorruptionError::Block)
    }

    pub(crate) fn first_key(&self) -> &[u8] {
        debug_assert!(self.record_count > 0);
        let start = KEY_PREFIX_LEN_LEN;
        &self.metadata()[start..start + self.key_len]
    }

    pub(crate) fn last_key(&self) -> &[u8] {
        if let Some(last_key) = &self.last_key {
            return last_key;
        }
        let start = KEY_PREFIX_LEN_LEN + (self.record_count - 1) * self.key_len;
        &self.metadata()[start..start + self.key_len]
    }

    pub(crate) fn record_count(&self) -> usize {
        self.record_count
    }

    pub(crate) fn key_at_index(&self, index: usize) -> Result<&[u8], CorruptionError> {
        self.key_at(index)
    }

    pub(crate) fn record_at_index_with_payload<'key, 'value>(
        &'key self,
        index: usize,
        payload: &'value [u8],
    ) -> Result<ParsedRecord<'key, 'value>, CorruptionError> {
        let key = self.key_at(index)?;
        let value = self.value_at_index_with_payload(index, payload)?;
        Ok(ParsedRecord { key, value })
    }

    pub(crate) fn lower_bound_index(&self, key: &[u8]) -> usize {
        self.partition_point_by_key(key)
    }

    pub(crate) fn compare_key_at_index(&self, index: usize, key: &[u8]) -> Ordering {
        self.compare_key(index, key).unwrap_or(Ordering::Greater)
    }

    pub(crate) fn value_at_if_key_with_payload<'value>(
        &self,
        index: usize,
        key: &[u8],
        payload: &'value [u8],
    ) -> Option<&'value [u8]> {
        if !self.key_matches_at_index(index, key) {
            return None;
        }
        self.value_at_index_with_payload(index, payload).ok()
    }

    pub(crate) fn value_at_index<'a>(
        &'a self,
        index: usize,
        decoded_payload: Option<&'a [u8]>,
    ) -> Result<&'a [u8], CorruptionError> {
        let payload = self.payload_bytes(decoded_payload)?;
        self.value_at_index_with_payload(index, payload)
    }

    pub(crate) fn value_at_index_with_payload<'value>(
        &self,
        index: usize,
        payload: &'value [u8],
    ) -> Result<&'value [u8], CorruptionError> {
        let range = self.value_region.range(self.metadata(), index)?;
        payload.get(range).ok_or(CorruptionError::Block)
    }

    pub(crate) fn key_matches_at_index(&self, index: usize, key: &[u8]) -> bool {
        self.compare_key(index, key).ok() == Some(Ordering::Equal)
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
            let start = KEY_PREFIX_LEN_LEN
                .checked_add(
                    index
                        .checked_mul(self.key_len)
                        .ok_or(CorruptionError::Block)?,
                )
                .ok_or(CorruptionError::Block)?;
            let end = start
                .checked_add(self.key_len)
                .ok_or(CorruptionError::Block)?;
            if end > self.key_section_len {
                return Err(CorruptionError::Block);
            }
            return Ok(&self.metadata()[start..end]);
        }

        let start = index
            .checked_mul(self.key_len)
            .ok_or(CorruptionError::Block)?;
        let end = start
            .checked_add(self.key_len)
            .ok_or(CorruptionError::Block)?;
        let full_keys = self.full_keys.get_or_init(|| self.reconstruct_full_keys());
        if end > full_keys.len() {
            return Err(CorruptionError::Block);
        }
        Ok(&full_keys[start..end])
    }

    fn metadata(&self) -> &[u8] {
        &self.bytes
    }

    pub(crate) fn payload_bytes<'a>(
        &'a self,
        decoded_payload: Option<&'a [u8]>,
    ) -> Result<&'a [u8], CorruptionError> {
        if !self.payload_is_borrowable {
            let payload = decoded_payload.ok_or(CorruptionError::Block)?;
            if payload.len() != self.payload_len {
                return Err(CorruptionError::Block);
            }
            return Ok(payload);
        }
        let payload_end = self
            .payload_data_offset
            .checked_add(self.payload_len)
            .ok_or(CorruptionError::Block)?;
        self.bytes
            .get(self.payload_data_offset..payload_end)
            .ok_or(CorruptionError::Block)
    }

    fn compare_key(&self, index: usize, key: &[u8]) -> Result<Ordering, CorruptionError> {
        if index >= self.record_count || key.len() != self.key_len {
            return Err(CorruptionError::Block);
        }
        let prefix_start = KEY_PREFIX_LEN_LEN;
        let prefix_end = prefix_start
            .checked_add(self.key_prefix_len)
            .ok_or(CorruptionError::Block)?;
        let prefix = self
            .metadata()
            .get(prefix_start..prefix_end)
            .ok_or(CorruptionError::Block)?;
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
        let suffix_table_start = KEY_PREFIX_LEN_LEN
            .checked_add(self.key_prefix_len)
            .ok_or(CorruptionError::Block)?;
        let start = suffix_table_start
            .checked_add(
                index
                    .checked_mul(self.suffix_len)
                    .ok_or(CorruptionError::Block)?,
            )
            .ok_or(CorruptionError::Block)?;
        let end = start
            .checked_add(self.suffix_len)
            .ok_or(CorruptionError::Block)?;
        if end > self.key_section_len {
            return Err(CorruptionError::Block);
        }
        Ok(&self.metadata()[start..end])
    }

    fn needs_full_key_buffer(&self) -> bool {
        self.key_prefix_len > 0 && self.record_count > 1
    }

    fn reconstruct_full_keys(&self) -> Vec<u8> {
        let prefix_start = KEY_PREFIX_LEN_LEN;
        let prefix_end = prefix_start + self.key_prefix_len;
        let prefix = &self.metadata()[prefix_start..prefix_end];
        let suffix_table_start = prefix_end;
        let mut full_keys = Vec::with_capacity(self.record_count * self.key_len);
        for index in 0..self.record_count {
            let suffix_start = suffix_table_start + index * self.suffix_len;
            let suffix_end = suffix_start + self.suffix_len;
            full_keys.extend_from_slice(prefix);
            full_keys.extend_from_slice(&self.metadata()[suffix_start..suffix_end]);
        }
        full_keys
    }

    fn validate_key_ordering(
        &self,
        entry: &BlockIndexEntry,
        upper_key_bound: BlockKeyUpperBound<'_>,
    ) -> Result<(), CorruptionError> {
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
        let last_key = self.last_key();
        let upper_key = upper_key_bound.key();
        if upper_key.len() != self.key_len {
            return Err(CorruptionError::Block);
        }
        let in_bounds = match upper_key_bound {
            BlockKeyUpperBound::Exclusive(_) => last_key < upper_key,
            BlockKeyUpperBound::Inclusive(_) => last_key <= upper_key,
        };
        if !in_bounds {
            return Err(CorruptionError::Block);
        }
        Ok(())
    }
}

impl<'a> BlockKeyUpperBound<'a> {
    fn key(self) -> &'a [u8] {
        match self {
            Self::Exclusive(key) | Self::Inclusive(key) => key,
        }
    }
}

/// Block layout derived from the lookup metadata during decoding.
struct BlockLayout {
    record_count: usize,
    key_len: usize,
    suffix_len: usize,
    lookup: BlockLookupLayout,
    lookup_metadata_len: usize,
    payload_offset: usize,
    payload_len: usize,
    payload_data_offset: usize,
    payload_is_borrowable: bool,
    value_checksum_offset: usize,
    block_checksum: BlockChecksumKind,
    value_payload_compression: ValuePayloadCompressionKind,
    value_region: BlockValueRegion,
    block_len: usize,
}

impl BlockLayout {
    fn from_block_bytes(
        bytes: &[u8],
        entry: &BlockIndexEntry,
        key_len: usize,
        value_layout: ValueLayout,
        block_checksum: BlockChecksumKind,
        value_payload_compression: ValuePayloadCompressionKind,
    ) -> Result<Self, CorruptionError> {
        if bytes.len() != entry.block_len as usize {
            return Err(CorruptionError::Block);
        }
        let mut layout = Self::from_metadata_prefix(
            bytes,
            entry,
            key_len,
            value_layout,
            block_checksum,
            value_payload_compression,
        )?;
        layout.read_payload_frame(bytes)?;
        Ok(layout)
    }

    fn from_metadata_prefix(
        bytes: &[u8],
        entry: &BlockIndexEntry,
        key_len: usize,
        value_layout: ValueLayout,
        block_checksum: BlockChecksumKind,
        value_payload_compression: ValuePayloadCompressionKind,
    ) -> Result<Self, CorruptionError> {
        let record_count = entry.record_count as usize;
        if record_count == 0 {
            return Err(CorruptionError::Block);
        }
        let key_prefix_len = BlockLookupLayout::read_key_prefix_len(bytes)?;
        let lookup = BlockLookupLayout::new(record_count, key_len, value_layout, key_prefix_len)?;
        let lookup_metadata_len = lookup.lookup_metadata_len;
        let metadata_with_checksum_len = lookup.metadata_with_checksum_len(block_checksum)?;
        if metadata_with_checksum_len > bytes.len() {
            return Err(CorruptionError::Block);
        }
        let payload_offset = metadata_with_checksum_len;
        let metadata = &bytes[..lookup_metadata_len];
        let value_region = BlockValueRegion::from_metadata(
            value_layout,
            record_count,
            lookup.key_section_len,
            payload_offset,
            metadata,
        )
        .ok_or(CorruptionError::Block)?;
        let payload_len = value_region.payload_len();
        if payload_offset > entry.block_len as usize {
            return Err(CorruptionError::Block);
        }
        Ok(Self {
            record_count,
            key_len,
            suffix_len: key_len - key_prefix_len,
            lookup,
            lookup_metadata_len,
            payload_offset,
            payload_len,
            payload_data_offset: payload_offset + value_payload_compression.frame_header_len(),
            payload_is_borrowable: value_payload_compression == ValuePayloadCompressionKind::None,
            value_checksum_offset: payload_offset,
            block_checksum,
            value_payload_compression,
            value_region,
            block_len: entry.block_len as usize,
        })
    }

    fn read_payload_frame(&mut self, bytes: &[u8]) -> Result<(), CorruptionError> {
        let frame_header = bytes
            .get(self.payload_offset..)
            .ok_or(CorruptionError::Block)?;
        let frame = self
            .value_payload_compression
            .parse_frame_header(frame_header, self.payload_len)?;
        self.payload_data_offset = self
            .payload_offset
            .checked_add(self.value_payload_compression.frame_header_len())
            .ok_or(CorruptionError::Block)?;
        self.value_checksum_offset = self
            .payload_offset
            .checked_add(frame.frame_len())
            .ok_or(CorruptionError::Block)?;
        let value_checksum_end = self
            .value_checksum_offset
            .checked_add(self.block_checksum.digest_len())
            .ok_or(CorruptionError::Block)?;
        if value_checksum_end > self.block_len {
            return Err(CorruptionError::Block);
        }
        self.payload_is_borrowable = frame.is_raw_borrowable();
        Ok(())
    }

    fn verify_lookup_checksum(&self, bytes: &[u8]) -> Result<(), CorruptionError> {
        let stored_checksum =
            read_stored_checksum(bytes, self.lookup_metadata_len, self.block_checksum)?;
        let checked = bytes
            .get(..self.lookup_metadata_len)
            .ok_or(CorruptionError::Block)?;
        if !self.block_checksum.verify(checked, stored_checksum) {
            return Err(CorruptionError::Block);
        }
        Ok(())
    }

    fn verify_payload_checksum(&self, bytes: &[u8]) -> Result<(), CorruptionError> {
        let stored_checksum =
            read_stored_checksum(bytes, self.value_checksum_offset, self.block_checksum)?;
        let checked = bytes
            .get(self.payload_offset..self.value_checksum_offset)
            .ok_or(CorruptionError::Block)?;
        if !self.block_checksum.verify(checked, stored_checksum) {
            return Err(CorruptionError::Block);
        }
        Ok(())
    }

    fn validate_key_section(&self, bytes: &[u8]) -> Result<(), CorruptionError> {
        if self.lookup.key_section_len > bytes.len() {
            return Err(CorruptionError::Block);
        }
        Ok(())
    }

    fn materialize_last_key_if_needed(
        &self,
        bytes: &[u8],
    ) -> Result<Option<Vec<u8>>, CorruptionError> {
        if self.lookup.key_prefix_len == 0 || self.record_count == 1 {
            return Ok(None);
        }
        let index = self.record_count - 1;
        let prefix_start = KEY_PREFIX_LEN_LEN;
        let prefix_end = prefix_start
            .checked_add(self.lookup.key_prefix_len)
            .ok_or(CorruptionError::Block)?;
        let suffix_start = prefix_end
            .checked_add(
                index
                    .checked_mul(self.suffix_len)
                    .ok_or(CorruptionError::Block)?,
            )
            .ok_or(CorruptionError::Block)?;
        let suffix_end = suffix_start
            .checked_add(self.suffix_len)
            .ok_or(CorruptionError::Block)?;
        if suffix_end > self.lookup.key_section_len {
            return Err(CorruptionError::Block);
        }
        let mut key = Vec::with_capacity(self.key_len);
        key.extend_from_slice(&bytes[prefix_start..prefix_end]);
        key.extend_from_slice(&bytes[suffix_start..suffix_end]);
        Ok(Some(key))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{
        BlockChecksumKind, ValuePayloadEncoder,
        block::BlockBuilder,
        record::{EntryRef, EntrySource},
    };
    #[cfg(any(feature = "value-compression-lz4", feature = "value-compression-zstd"))]
    use crate::format::{ValuePayloadCompressionPolicy, ValuePayloadDecoder};

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

    #[cfg(any(feature = "value-compression-lz4", feature = "value-compression-zstd"))]
    fn block_builder<'a>(
        entries: &'a Entries<'a>,
        key_len: usize,
        value_layout: ValueLayout,
        checksum: BlockChecksumKind,
        compression: ValuePayloadCompressionKind,
    ) -> BlockBuilder<'a, Entries<'a>> {
        BlockBuilder::new(
            entries,
            key_len,
            value_layout,
            checksum,
            compression,
            ValuePayloadCompressionPolicy::DEFAULT,
        )
    }

    #[cfg(not(any(feature = "value-compression-lz4", feature = "value-compression-zstd")))]
    fn block_builder<'a>(
        entries: &'a Entries<'a>,
        key_len: usize,
        value_layout: ValueLayout,
        checksum: BlockChecksumKind,
        compression: ValuePayloadCompressionKind,
    ) -> BlockBuilder<'a, Entries<'a>> {
        BlockBuilder::new(entries, key_len, value_layout, checksum, compression)
    }

    mod prefix_stripping {
        use super::*;

        #[test]
        fn stores_common_prefix_once_and_round_trips_full_keys() {
            let entries = Entries {
                entries: &[(b"aa01", b"first"), (b"aa02", b"second")],
            };
            let checksum = BlockChecksumKind::None;
            let compression = ValuePayloadCompressionKind::None;
            let mut encoder = ValuePayloadEncoder::new(compression);
            let block = block_builder(&entries, 4, ValueLayout::VARIABLE, checksum, compression)
                .encode(&mut encoder)
                .expect("block should encode");
            let entry = BlockIndexEntry {
                first_key: b"aa01".to_vec(),
                block_offset: 0,
                block_len: u32::try_from(block.len()).expect("block len"),
                record_count: 2,
            };

            assert_eq!(
                u32::from_le_bytes(block[..4].try_into().expect("prefix len")),
                3
            );
            assert_eq!(&block[4..7], b"aa0");
            assert_eq!(&block[7..9], b"12");

            let decoded = DecodedBlock::decode(block, &entry, BlockDecodeOptions {
                key_len: 4,
                value_layout: ValueLayout::VARIABLE,
                block_checksum: checksum,
                value_payload_compression: compression,
                verify_checksum: true,
                upper_key_bound: BlockKeyUpperBound::Inclusive(b"aa02"),
            })
            .expect("block should decode");

            assert!(
                decoded.full_keys.get().is_none(),
                "lookup methods should not materialize full key table"
            );
            assert_eq!(decoded.first_key(), b"aa01");
            assert_eq!(decoded.last_key(), b"aa02");
            let payload = decoded
                .payload_bytes(None)
                .expect("raw payload should be borrowable");
            let index = decoded.lower_bound_index(b"aa02");
            assert_eq!(
                decoded.value_at_if_key_with_payload(index, b"aa02", payload),
                Some(&b"second"[..])
            );
            assert!(
                decoded.full_keys.get().is_none(),
                "key lookup should compare prefix-stripped keys without materialization"
            );
            assert_eq!(
                decoded
                    .record_at_index_with_payload(1, payload)
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
        fn single_record_block_borrows_raw_key_section() {
            let entries = Entries {
                entries: &[(b"aa01", b"only")],
            };
            let checksum = BlockChecksumKind::None;
            let compression = ValuePayloadCompressionKind::None;
            let mut encoder = ValuePayloadEncoder::new(compression);
            let block = block_builder(&entries, 4, ValueLayout::VARIABLE, checksum, compression)
                .encode(&mut encoder)
                .expect("block should encode");
            let entry = BlockIndexEntry {
                first_key: b"aa01".to_vec(),
                block_offset: 0,
                block_len: u32::try_from(block.len()).expect("block len"),
                record_count: 1,
            };

            let decoded = DecodedBlock::decode(block, &entry, BlockDecodeOptions {
                key_len: 4,
                value_layout: ValueLayout::VARIABLE,
                block_checksum: checksum,
                value_payload_compression: compression,
                verify_checksum: true,
                upper_key_bound: BlockKeyUpperBound::Inclusive(b"aa01"),
            })
            .expect("block should decode");

            assert!(decoded.full_keys.get().is_none());
            assert_eq!(decoded.first_key(), b"aa01");
            assert_eq!(decoded.last_key(), b"aa01");
            let payload = decoded
                .payload_bytes(None)
                .expect("raw payload should be borrowable");
            let index = decoded.lower_bound_index(b"aa01");
            assert_eq!(
                decoded.value_at_if_key_with_payload(index, b"aa01", payload),
                Some(&b"only"[..])
            );
        }

        #[test]
        fn rejects_keys_outside_block_upper_bound() {
            let entries = Entries {
                entries: &[(b"aa01", b"first"), (b"aa02", b"second")],
            };
            let checksum = BlockChecksumKind::None;
            let compression = ValuePayloadCompressionKind::None;
            let mut encoder = ValuePayloadEncoder::new(compression);
            let block = block_builder(&entries, 4, ValueLayout::VARIABLE, checksum, compression)
                .encode(&mut encoder)
                .expect("block should encode");
            let entry = BlockIndexEntry {
                first_key: b"aa01".to_vec(),
                block_offset: 0,
                block_len: u32::try_from(block.len()).expect("block len"),
                record_count: 2,
            };
            let exclusive_upper = DecodedBlock::decode(block.clone(), &entry, BlockDecodeOptions {
                key_len: 4,
                value_layout: ValueLayout::VARIABLE,
                block_checksum: checksum,
                value_payload_compression: compression,
                verify_checksum: true,
                upper_key_bound: BlockKeyUpperBound::Exclusive(b"aa02"),
            });
            assert!(matches!(exclusive_upper, Err(CorruptionError::Block)));

            let inclusive_upper = DecodedBlock::decode(block, &entry, BlockDecodeOptions {
                key_len: 4,
                value_layout: ValueLayout::VARIABLE,
                block_checksum: checksum,
                value_payload_compression: compression,
                verify_checksum: true,
                upper_key_bound: BlockKeyUpperBound::Inclusive(b"aa01"),
            });
            assert!(matches!(inclusive_upper, Err(CorruptionError::Block)));
        }
    }

    #[cfg(any(feature = "value-compression-lz4", feature = "value-compression-zstd"))]
    mod payload_materialization {
        use super::*;

        #[cfg(feature = "value-compression-lz4")]
        #[test]
        fn lz4_payload_is_decoded_only_for_value_access() {
            assert_compressed_payload_decodes_lazily(ValuePayloadCompressionKind::Lz4);
        }

        #[cfg(feature = "value-compression-zstd")]
        #[test]
        fn zstd_payload_is_decoded_only_for_value_access() {
            assert_compressed_payload_decodes_lazily(ValuePayloadCompressionKind::ZstdLevel1);
        }

        fn assert_compressed_payload_decodes_lazily(compression: ValuePayloadCompressionKind) {
            let value = vec![7u8; 128 * 1024];
            let entries_data = [(b"aa01".as_slice(), value.as_slice())];
            let entries = Entries {
                entries: &entries_data,
            };
            let checksum = BlockChecksumKind::None;
            let policy =
                ValuePayloadCompressionPolicy::new(0, 1).expect("compression policy is valid");
            let mut encoder = ValuePayloadEncoder::new(compression);
            let block = BlockBuilder::new(
                &entries,
                4,
                ValueLayout::VARIABLE,
                checksum,
                compression,
                policy,
            )
            .encode(&mut encoder)
            .expect("block should encode");
            let entry = BlockIndexEntry {
                first_key: b"aa01".to_vec(),
                block_offset: 0,
                block_len: u32::try_from(block.len()).expect("block len"),
                record_count: 1,
            };

            let decoded = DecodedBlock::decode(block, &entry, BlockDecodeOptions {
                key_len: 4,
                value_layout: ValueLayout::VARIABLE,
                block_checksum: checksum,
                value_payload_compression: compression,
                verify_checksum: true,
                upper_key_bound: BlockKeyUpperBound::Inclusive(b"aa01"),
            })
            .expect("block metadata should decode without materializing payload");
            assert!(
                matches!(decoded.value_at_index(0, None), Err(CorruptionError::Block)),
                "compressed values require caller-owned decoded payload scratch"
            );

            let mut decoder = ValuePayloadDecoder::new(compression);
            let decoded_payload = decoded
                .decode_payload_if_needed(&mut decoder)
                .expect("payload should decode")
                .expect("payload should be stored compressed");
            assert_eq!(
                decoded
                    .value_at_index(0, Some(decoded_payload.as_slice()))
                    .expect("value should borrow from decoded payload"),
                value.as_slice()
            );
        }
    }
}
