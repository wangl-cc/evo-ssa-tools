//! Block encoder: sorted entries in, on-disk block bytes out.

use super::layout::{BlockLookupLayout, BlockValueRegion, CHECKSUM_LEN, block_crc};
use crate::format::{FormatError, ValueLayout, common_prefix_len, record::EntrySource};

/// Encodes one sorted run of key/value entries into the on-disk block layout.
pub(crate) struct BlockBuilder<'a, S: EntrySource + ?Sized> {
    entries: &'a S,
    key_len: usize,
    value_layout: ValueLayout,
    target_block_size: usize,
}

impl<'a, S: EntrySource + ?Sized> BlockBuilder<'a, S> {
    pub(crate) fn new(
        entries: &'a S,
        key_len: usize,
        value_layout: ValueLayout,
        target_block_size: usize,
    ) -> Self {
        Self {
            entries,
            key_len,
            value_layout,
            target_block_size,
        }
    }

    pub(crate) fn encode(&self) -> Result<Vec<u8>, FormatError> {
        let record_count = self.entries.len();
        let key_prefix_len = self.common_key_prefix_len();
        let lookup_layout = BlockLookupLayout::new(
            record_count,
            self.key_len,
            self.value_layout,
            key_prefix_len,
        )
        .map_err(|_| FormatError::limit("block lookup metadata length"))?;
        let payload_len = (0..record_count).try_fold(0usize, |len, index| {
            len.checked_add(self.entries.entry(index).value().len())
                .ok_or(FormatError::limit("block value payload length"))
        })?;
        let payload_offset = lookup_layout
            .metadata_with_crc_len()
            .map_err(|_| FormatError::limit("block payload offset"))?;
        let value_region = BlockValueRegion::for_write(
            self.value_layout,
            record_count,
            lookup_layout.key_section_len,
            payload_offset,
            payload_len,
        )
        .ok_or(FormatError::limit("block value region offsets"))?;
        let encoded_len = value_region
            .payload_offset()
            .checked_add(payload_len)
            .and_then(|len| len.checked_add(CHECKSUM_LEN))
            .ok_or(FormatError::limit("block length"))?;
        let block_len = encoded_len.max(self.target_block_size);
        let mut block = Vec::with_capacity(block_len);
        self.write_keys(&mut block, key_prefix_len);
        value_region.write_index(self.entries, &mut block)?;
        debug_assert_eq!(block.len(), lookup_layout.lookup_metadata_len);
        let lookup_crc = block_crc(&block);
        block.extend_from_slice(&lookup_crc.to_le_bytes());
        debug_assert_eq!(block.len(), value_region.payload_offset());
        self.write_values(&mut block);
        let value_crc = block_crc(&block[value_region.payload_offset()..]);
        block.extend_from_slice(&value_crc.to_le_bytes());
        if block.len() < block_len {
            block.resize(block_len, 0);
        }
        debug_assert_eq!(block.len(), block_len);
        Ok(block)
    }

    fn common_key_prefix_len(&self) -> usize {
        if self.entries.is_empty() {
            return 0;
        }
        common_prefix_len(
            self.entries.entry(0).key(),
            self.entries.entry(self.entries.len() - 1).key(),
        )
    }

    fn write_keys(&self, block: &mut Vec<u8>, prefix_len: usize) {
        let prefix_len =
            u32::try_from(prefix_len).expect("key prefix length is bounded by key_len");
        block.extend_from_slice(&prefix_len.to_le_bytes());
        if self.entries.is_empty() {
            return;
        }
        let prefix_len = prefix_len as usize;
        block.extend_from_slice(&self.entries.entry(0).key()[..prefix_len]);
        for index in 0..self.entries.len() {
            block.extend_from_slice(&self.entries.entry(index).key()[prefix_len..]);
        }
    }

    fn write_values(&self, block: &mut Vec<u8>) {
        for index in 0..self.entries.len() {
            block.extend_from_slice(self.entries.entry(index).value());
        }
    }
}
