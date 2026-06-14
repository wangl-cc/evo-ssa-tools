//! Block encoder: sorted entries in, on-disk block bytes out.

use super::layout::{BLOCK_FOOTER_LEN, BlockFooter, BlockValueRegion};
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
        let suffix_len = self.key_len - key_prefix_len;
        let key_region_len = key_prefix_len
            .checked_add(
                record_count
                    .checked_mul(suffix_len)
                    .ok_or(FormatError::limit("block key region length"))?,
            )
            .ok_or(FormatError::limit("block key region length"))?;
        let payload_len = (0..record_count).try_fold(0usize, |len, index| {
            len.checked_add(self.entries.entry(index).value().len())
                .ok_or(FormatError::limit("block value payload length"))
        })?;
        let value_region = BlockValueRegion::for_write(
            self.value_layout,
            record_count,
            key_region_len,
            payload_len,
        )
        .ok_or(FormatError::limit("block value region offsets"))?;
        let logical_len = value_region
            .payload_offset()
            .checked_add(payload_len)
            .ok_or(FormatError::limit("block logical length"))?;
        let encoded_len = logical_len
            .checked_add(BLOCK_FOOTER_LEN)
            .ok_or(FormatError::limit("block length"))?;
        let block_len = encoded_len.max(self.target_block_size);
        let mut block = Vec::with_capacity(block_len);
        self.write_keys(&mut block, key_prefix_len);
        value_region.write_index(self.entries, &mut block)?;
        self.write_values(&mut block);
        Self::finish_block(block, block_len, key_prefix_len, value_region)
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
        if self.entries.is_empty() {
            return;
        }
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

    fn finish_block(
        mut block: Vec<u8>,
        block_len: usize,
        key_prefix_len: usize,
        value_region: BlockValueRegion,
    ) -> Result<Vec<u8>, FormatError> {
        debug_assert!(block_len >= BLOCK_FOOTER_LEN);
        let body_len = block_len - BLOCK_FOOTER_LEN;
        if block.len() < body_len {
            block.resize(body_len, 0);
        }
        let footer = BlockFooter::new(key_prefix_len, value_region, &block)?;
        footer.into_bytes(&mut block)?;
        debug_assert_eq!(block.len(), block_len);
        Ok(block)
    }
}
