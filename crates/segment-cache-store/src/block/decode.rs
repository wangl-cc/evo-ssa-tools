//! Block decoder: on-disk block bytes in, borrowed record views out.
//!
//! Decoding validates the block lookup metadata, value layout, and key
//! ordering before any record is returned. A block that fails validation is
//! reported as [`CorruptionError::Block`] and degrades to a cache miss.

use std::cmp::Ordering;

#[cfg(feature = "value-compression")]
use super::ValuePayloadDecoder;
#[cfg(feature = "block-checksum")]
use super::layout::read_stored_checksum;
use super::{
    BlockChecksumKind, ValuePayloadCompressionKind,
    layout::{BLOCK_METADATA_HEADER_LEN, BlockLookupLayout, BlockValueRegion},
};
use crate::{error::CorruptionError, value::ValueLayout};

/// Raw block bytes plus derived offsets needed for borrowed record access.
///
/// Prefix-compressed keys remain split in the block. Scan cursors reconstruct
/// only their current key into caller-owned scratch, while compressed values
/// borrow from a decoder-owned buffer supplied by the read session.
#[derive(Debug)]
pub(crate) struct DecodedBlock {
    bytes: Vec<u8>,
    record_count: usize,
    key_len: usize,
    key_prefix_len: usize,
    suffix_len: usize,
    key_section_len: usize,
    #[cfg(any(feature = "block-checksum", feature = "value-compression"))]
    payload_offset: usize,
    payload_len: usize,
    payload_data_offset: usize,
    #[cfg(any(feature = "block-checksum", feature = "value-compression"))]
    value_checksum_offset: usize,
    #[cfg(feature = "value-compression")]
    payload_is_borrowable: bool,
    #[cfg(feature = "value-compression")]
    value_payload_compression: ValuePayloadCompressionKind,
    #[cfg(feature = "block-checksum")]
    block_checksum: BlockChecksumKind,
    value_region: BlockValueRegion,
}

/// Borrowed key/value record decoded from stored bytes.
pub(crate) struct ParsedRecord<'key, 'value> {
    pub(crate) key: &'key [u8],
    pub(crate) value: &'value [u8],
}

#[derive(Clone, Copy)]
pub(crate) struct BlockKeyRangeRef<'a> {
    pub(crate) prefix: &'a [u8],
    pub(crate) min_suffix: &'a [u8],
    pub(crate) max_suffix: &'a [u8],
}

#[derive(Clone, Copy)]
pub(crate) struct BlockDecodeOptions<'a> {
    pub(crate) expected_key_range: BlockKeyRangeRef<'a>,
    pub(crate) key_len: usize,
    pub(crate) value_layout: ValueLayout,
    pub(crate) block_checksum: BlockChecksumKind,
    pub(crate) value_payload_compression: ValuePayloadCompressionKind,
    #[cfg(feature = "block-checksum")]
    pub(crate) verify_lookup_checksum: bool,
}

impl DecodedBlock {
    pub(crate) fn decode(
        bytes: Vec<u8>,
        options: BlockDecodeOptions<'_>,
    ) -> Result<Self, CorruptionError> {
        let layout = BlockLayout::from_block_bytes(
            &bytes,
            options.key_len,
            options.value_layout,
            options.block_checksum,
            options.value_payload_compression,
        )?;
        #[cfg(feature = "block-checksum")]
        if options.verify_lookup_checksum {
            layout.verify_lookup_checksum(&bytes)?;
        }
        Self::from_bytes(bytes, layout, options.expected_key_range)
    }

    fn from_bytes(
        bytes: Vec<u8>,
        layout: BlockLayout,
        expected_key_range: BlockKeyRangeRef<'_>,
    ) -> Result<Self, CorruptionError> {
        layout.validate_key_section(&bytes)?;
        let block = Self {
            bytes,
            record_count: layout.record_count,
            key_len: layout.key_len,
            key_prefix_len: layout.lookup.key_prefix_len,
            suffix_len: layout.suffix_len,
            key_section_len: layout.lookup.key_section_len,
            #[cfg(any(feature = "block-checksum", feature = "value-compression"))]
            payload_offset: layout.payload_offset,
            payload_len: layout.payload_len,
            payload_data_offset: layout.payload_data_offset,
            #[cfg(any(feature = "block-checksum", feature = "value-compression"))]
            value_checksum_offset: layout.value_checksum_offset,
            #[cfg(feature = "value-compression")]
            payload_is_borrowable: layout.payload_is_borrowable,
            #[cfg(feature = "value-compression")]
            value_payload_compression: layout.value_payload_compression,
            #[cfg(feature = "block-checksum")]
            block_checksum: layout.block_checksum,
            value_region: layout.value_region,
        };
        block.value_region.validate(block.metadata())?;
        block.validate_key_ordering(expected_key_range)?;
        Ok(block)
    }

    #[cfg(feature = "value-compression")]
    pub(crate) fn decode_payload_if_needed(
        &self,
        payload_decoder: &mut ValuePayloadDecoder,
    ) -> Result<Option<Vec<u8>>, CorruptionError> {
        if self.payload_is_borrowable {
            return Ok(None);
        }
        self.value_payload_compression.decode_frame(
            payload_decoder,
            self.payload_frame()?,
            self.payload_len,
        )
    }

    pub(crate) fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    #[cfg(feature = "block-checksum")]
    pub(crate) fn verify_payload_checksum(&self) -> Result<(), CorruptionError> {
        let stored_checksum =
            read_stored_checksum(&self.bytes, self.value_checksum_offset, self.block_checksum)?;
        let checked = self
            .bytes
            .get(self.payload_offset..self.value_checksum_offset)
            .ok_or(CorruptionError::Block)?;
        if !self.block_checksum.verify(checked, stored_checksum) {
            return Err(CorruptionError::Block);
        }
        Ok(())
    }

    #[cfg(feature = "value-compression")]
    pub(crate) fn payload_frame(&self) -> Result<&[u8], CorruptionError> {
        self.bytes
            .get(self.payload_offset..self.value_checksum_offset)
            .ok_or(CorruptionError::Block)
    }

    pub(crate) fn record_count(&self) -> usize {
        self.record_count
    }

    pub(crate) fn write_key_at_index(
        &self,
        index: usize,
        destination: &mut [u8],
    ) -> Result<(), CorruptionError> {
        if destination.len() != self.key_len {
            return Err(CorruptionError::Block);
        }
        let prefix = self.key_prefix()?;
        let suffix = self.key_suffix_at(index)?;
        let (destination_prefix, destination_suffix) = destination.split_at_mut(prefix.len());
        destination_prefix.copy_from_slice(prefix);
        destination_suffix.copy_from_slice(suffix);
        Ok(())
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

    #[cfg(feature = "value-compression")]
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

    fn metadata(&self) -> &[u8] {
        &self.bytes
    }

    #[cfg(feature = "value-compression")]
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

    #[cfg(not(feature = "value-compression"))]
    pub(crate) fn payload_bytes(&self) -> Result<&[u8], CorruptionError> {
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
        let prefix_start = BLOCK_METADATA_HEADER_LEN;
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
        let suffix_table_start = BLOCK_METADATA_HEADER_LEN
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

    fn validate_key_ordering(&self, expected: BlockKeyRangeRef<'_>) -> Result<(), CorruptionError> {
        if self.key_prefix()? != expected.prefix
            || self.key_suffix_at(0)? != expected.min_suffix
            || self.key_suffix_at(self.record_count - 1)? != expected.max_suffix
        {
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

    fn key_prefix(&self) -> Result<&[u8], CorruptionError> {
        self.metadata()
            .get(BLOCK_METADATA_HEADER_LEN..BLOCK_METADATA_HEADER_LEN + self.key_prefix_len)
            .ok_or(CorruptionError::Block)
    }
}

/// Block layout derived from the lookup metadata during decoding.
struct BlockLayout {
    record_count: usize,
    key_len: usize,
    suffix_len: usize,
    lookup: BlockLookupLayout,
    #[cfg(feature = "block-checksum")]
    lookup_metadata_len: usize,
    payload_offset: usize,
    payload_len: usize,
    payload_data_offset: usize,
    #[cfg(feature = "value-compression")]
    payload_is_borrowable: bool,
    value_checksum_offset: usize,
    block_checksum: BlockChecksumKind,
    #[cfg(feature = "value-compression")]
    value_payload_compression: ValuePayloadCompressionKind,
    value_region: BlockValueRegion,
    block_len: usize,
}

impl BlockLayout {
    fn from_block_bytes(
        bytes: &[u8],
        key_len: usize,
        value_layout: ValueLayout,
        block_checksum: BlockChecksumKind,
        value_payload_compression: ValuePayloadCompressionKind,
    ) -> Result<Self, CorruptionError> {
        let mut layout = Self::from_metadata_prefix(
            bytes,
            key_len,
            value_layout,
            block_checksum,
            value_payload_compression,
        )?;
        #[cfg(feature = "value-compression")]
        layout.read_payload_frame(bytes)?;
        #[cfg(not(feature = "value-compression"))]
        layout.read_raw_payload()?;
        Ok(layout)
    }

    fn from_metadata_prefix(
        bytes: &[u8],
        key_len: usize,
        value_layout: ValueLayout,
        block_checksum: BlockChecksumKind,
        value_payload_compression: ValuePayloadCompressionKind,
    ) -> Result<Self, CorruptionError> {
        let record_count = BlockLookupLayout::read_record_count(bytes)?;
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
        if payload_offset > bytes.len() {
            return Err(CorruptionError::Block);
        }
        #[cfg(feature = "value-compression")]
        let payload_data_offset = payload_offset
            .checked_add(value_payload_compression.frame_header_len())
            .ok_or(CorruptionError::Block)?;
        #[cfg(not(feature = "value-compression"))]
        let payload_data_offset = {
            if value_payload_compression != ValuePayloadCompressionKind::None {
                return Err(CorruptionError::Block);
            }
            payload_offset
        };
        Ok(Self {
            record_count,
            key_len,
            suffix_len: key_len - key_prefix_len,
            lookup,
            #[cfg(feature = "block-checksum")]
            lookup_metadata_len,
            payload_offset,
            payload_len,
            payload_data_offset,
            #[cfg(feature = "value-compression")]
            payload_is_borrowable: value_payload_compression == ValuePayloadCompressionKind::None,
            value_checksum_offset: payload_offset,
            block_checksum,
            #[cfg(feature = "value-compression")]
            value_payload_compression,
            value_region,
            block_len: bytes.len(),
        })
    }

    #[cfg(feature = "value-compression")]
    fn read_payload_frame(&mut self, bytes: &[u8]) -> Result<(), CorruptionError> {
        let value_checksum_offset = self
            .block_len
            .checked_sub(self.block_checksum.digest_len())
            .ok_or(CorruptionError::Block)?;
        let frame_bytes = bytes
            .get(self.payload_offset..value_checksum_offset)
            .ok_or(CorruptionError::Block)?;
        let frame = self
            .value_payload_compression
            .parse_frame(frame_bytes, self.payload_len)?;
        self.payload_data_offset = self
            .payload_offset
            .checked_add(self.value_payload_compression.frame_header_len())
            .ok_or(CorruptionError::Block)?;
        if frame.frame_len() != value_checksum_offset - self.payload_offset {
            return Err(CorruptionError::Block);
        }
        self.value_checksum_offset = value_checksum_offset;
        self.payload_is_borrowable = frame.is_raw_borrowable();
        Ok(())
    }

    #[cfg(not(feature = "value-compression"))]
    fn read_raw_payload(&mut self) -> Result<(), CorruptionError> {
        self.value_checksum_offset = self
            .payload_offset
            .checked_add(self.payload_len)
            .ok_or(CorruptionError::Block)?;
        let block_end = self
            .value_checksum_offset
            .checked_add(self.block_checksum.digest_len())
            .ok_or(CorruptionError::Block)?;
        if block_end != self.block_len {
            return Err(CorruptionError::Block);
        }
        Ok(())
    }

    #[cfg(feature = "block-checksum")]
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

    fn validate_key_section(&self, bytes: &[u8]) -> Result<(), CorruptionError> {
        if self.lookup.key_section_len > bytes.len() {
            return Err(CorruptionError::Block);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "value-compression")]
    use crate::block::{ValuePayloadCompressionPolicy, ValuePayloadEncoder};
    use crate::{
        block::{BlockBuilder, BlockChecksumKind},
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

    #[cfg(feature = "value-compression")]
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

    #[cfg(not(feature = "value-compression"))]
    fn block_builder<'a>(
        entries: &'a Entries<'a>,
        key_len: usize,
        value_layout: ValueLayout,
        checksum: BlockChecksumKind,
        compression: ValuePayloadCompressionKind,
    ) -> BlockBuilder<'a, Entries<'a>> {
        BlockBuilder::new(entries, key_len, value_layout, checksum, compression)
    }

    #[cfg(feature = "value-compression")]
    fn encode_block<S: EntrySource + ?Sized>(
        builder: BlockBuilder<'_, S>,
        compression: ValuePayloadCompressionKind,
    ) -> Vec<u8> {
        let mut encoder = ValuePayloadEncoder::new(compression);
        builder.encode(&mut encoder).expect("block should encode")
    }

    #[cfg(not(feature = "value-compression"))]
    fn encode_block<S: EntrySource + ?Sized>(
        builder: BlockBuilder<'_, S>,
        _compression: ValuePayloadCompressionKind,
    ) -> Vec<u8> {
        builder.encode().expect("block should encode")
    }

    #[cfg(feature = "value-compression")]
    fn raw_payload(block: &DecodedBlock) -> Result<&[u8], CorruptionError> {
        block.payload_bytes(None)
    }

    #[cfg(not(feature = "value-compression"))]
    fn raw_payload(block: &DecodedBlock) -> Result<&[u8], CorruptionError> {
        block.payload_bytes()
    }

    fn key_range<'a>(min_key: &'a [u8], max_key: &'a [u8]) -> BlockKeyRangeRef<'a> {
        let prefix_len = crate::key::common_prefix_len(min_key, max_key);
        BlockKeyRangeRef {
            prefix: &min_key[..prefix_len],
            min_suffix: &min_key[prefix_len..],
            max_suffix: &max_key[prefix_len..],
        }
    }

    mod prefix_stripping {
        use super::*;

        #[test]
        fn stores_common_prefix_once_and_reconstructs_into_reusable_scratch() {
            let entries = Entries {
                entries: &[(b"aa01", b"first"), (b"aa02", b"second")],
            };
            let checksum = BlockChecksumKind::None;
            let compression = ValuePayloadCompressionKind::None;
            let block = encode_block(
                block_builder(&entries, 4, ValueLayout::VARIABLE, checksum, compression),
                compression,
            );
            assert_eq!(
                u32::from_le_bytes(block[..4].try_into().expect("record count")),
                2
            );
            assert_eq!(
                u32::from_le_bytes(block[4..8].try_into().expect("prefix len")),
                3
            );
            assert_eq!(&block[8..11], b"aa0");
            assert_eq!(&block[11..13], b"12");

            let decoded = DecodedBlock::decode(block, BlockDecodeOptions {
                expected_key_range: key_range(b"aa01", b"aa02"),
                key_len: 4,
                value_layout: ValueLayout::VARIABLE,
                block_checksum: checksum,
                value_payload_compression: compression,
                #[cfg(feature = "block-checksum")]
                verify_lookup_checksum: true,
            })
            .expect("block should decode");

            let payload = raw_payload(&decoded).expect("raw payload should be borrowable");
            let index = decoded.lower_bound_index(b"aa02");
            assert_eq!(
                decoded.value_at_if_key_with_payload(index, b"aa02", payload),
                Some(&b"second"[..])
            );
            let mut key_scratch = vec![0; 4];
            decoded
                .write_key_at_index(0, &mut key_scratch)
                .expect("first key should reconstruct");
            assert_eq!(key_scratch, b"aa01");
            decoded
                .write_key_at_index(1, &mut key_scratch)
                .expect("scratch should be reusable for the last key");
            assert_eq!(key_scratch, b"aa02");
        }

        #[test]
        fn single_record_block_borrows_raw_key_section() {
            let entries = Entries {
                entries: &[(b"aa01", b"only")],
            };
            let checksum = BlockChecksumKind::None;
            let compression = ValuePayloadCompressionKind::None;
            let block = encode_block(
                block_builder(&entries, 4, ValueLayout::VARIABLE, checksum, compression),
                compression,
            );
            let decoded = DecodedBlock::decode(block, BlockDecodeOptions {
                expected_key_range: key_range(b"aa01", b"aa01"),
                key_len: 4,
                value_layout: ValueLayout::VARIABLE,
                block_checksum: checksum,
                value_payload_compression: compression,
                #[cfg(feature = "block-checksum")]
                verify_lookup_checksum: true,
            })
            .expect("block should decode");

            let mut key_scratch = vec![0; 4];
            decoded
                .write_key_at_index(0, &mut key_scratch)
                .expect("only key should reconstruct");
            assert_eq!(key_scratch, b"aa01");
            let payload = raw_payload(&decoded).expect("raw payload should be borrowable");
            let index = decoded.lower_bound_index(b"aa01");
            assert_eq!(
                decoded.value_at_if_key_with_payload(index, b"aa01", payload),
                Some(&b"only"[..])
            );
        }

        #[test]
        fn rejects_mismatched_index_range() {
            let entries = Entries {
                entries: &[(b"aa01", b"first"), (b"aa02", b"second")],
            };
            let checksum = BlockChecksumKind::None;
            let compression = ValuePayloadCompressionKind::None;
            let block = encode_block(
                block_builder(&entries, 4, ValueLayout::VARIABLE, checksum, compression),
                compression,
            );
            let short_range = DecodedBlock::decode(block.clone(), BlockDecodeOptions {
                expected_key_range: key_range(b"aa01", b"aa01"),
                key_len: 4,
                value_layout: ValueLayout::VARIABLE,
                block_checksum: checksum,
                value_payload_compression: compression,
                #[cfg(feature = "block-checksum")]
                verify_lookup_checksum: true,
            });
            assert!(matches!(short_range, Err(CorruptionError::Block)));

            let long_range = DecodedBlock::decode(block, BlockDecodeOptions {
                expected_key_range: key_range(b"aa01", b"aa03"),
                key_len: 4,
                value_layout: ValueLayout::VARIABLE,
                block_checksum: checksum,
                value_payload_compression: compression,
                #[cfg(feature = "block-checksum")]
                verify_lookup_checksum: true,
            });
            assert!(matches!(long_range, Err(CorruptionError::Block)));
        }
    }

    mod record_count {
        use super::*;

        #[test]
        fn zero_is_rejected_from_block_metadata() {
            let entries = Entries {
                entries: &[(b"key1", b"value")],
            };
            let checksum = BlockChecksumKind::None;
            let compression = ValuePayloadCompressionKind::None;
            let mut block = encode_block(
                block_builder(&entries, 4, ValueLayout::VARIABLE, checksum, compression),
                compression,
            );
            block[..4].copy_from_slice(&0u32.to_le_bytes());

            assert!(matches!(
                DecodedBlock::decode(block, BlockDecodeOptions {
                    expected_key_range: key_range(b"key1", b"key1"),
                    key_len: 4,
                    value_layout: ValueLayout::VARIABLE,
                    block_checksum: checksum,
                    value_payload_compression: compression,
                    #[cfg(feature = "block-checksum")]
                    verify_lookup_checksum: true,
                }),
                Err(CorruptionError::Block)
            ));
        }
    }

    #[cfg(any(feature = "checksum-crc32c", feature = "checksum-rapidhash"))]
    mod checksum_phases {
        use super::*;

        #[test]
        fn decode_verifies_lookup_before_payload() {
            let entries = Entries {
                entries: &[(b"aa01", b"first"), (b"aa02", b"second")],
            };
            let checksum = test_checksum();
            let compression = ValuePayloadCompressionKind::None;
            let mut bytes = encode_block(
                block_builder(&entries, 4, ValueLayout::VARIABLE, checksum, compression),
                compression,
            );
            let payload_checksum_byte = bytes
                .last_mut()
                .expect("checksummed block has a payload digest");
            *payload_checksum_byte ^= 0xff;

            let decoded = DecodedBlock::decode(bytes, BlockDecodeOptions {
                expected_key_range: key_range(b"aa01", b"aa02"),
                key_len: 4,
                value_layout: ValueLayout::VARIABLE,
                block_checksum: checksum,
                value_payload_compression: compression,
                verify_lookup_checksum: true,
            })
            .expect("payload corruption must not fail lookup decoding");

            assert!(matches!(
                decoded.verify_payload_checksum(),
                Err(CorruptionError::Block)
            ));
        }

        fn test_checksum() -> BlockChecksumKind {
            #[cfg(feature = "checksum-crc32c")]
            {
                BlockChecksumKind::Crc32c
            }
            #[cfg(all(not(feature = "checksum-crc32c"), feature = "checksum-rapidhash"))]
            {
                BlockChecksumKind::RapidHashV3_64
            }
        }
    }

    #[cfg(any(feature = "value-compression-lz4", feature = "value-compression-zstd"))]
    mod payload_materialization {
        use super::*;
        use crate::block::ValuePayloadDecoder;

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
            let decoded = DecodedBlock::decode(block, BlockDecodeOptions {
                expected_key_range: key_range(b"aa01", b"aa01"),
                key_len: 4,
                value_layout: ValueLayout::VARIABLE,
                block_checksum: checksum,
                value_payload_compression: compression,
                #[cfg(feature = "block-checksum")]
                verify_lookup_checksum: true,
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
