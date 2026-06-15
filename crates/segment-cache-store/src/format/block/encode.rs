//! Block encoder: sorted entries in, on-disk block bytes out.

use super::layout::{BlockLookupLayout, BlockValueRegion};
use crate::format::{
    BlockChecksumKind, FormatError, MAX_BLOCK_CHECKSUM_LEN, ValueLayout,
    ValuePayloadCompressionKind, common_prefix_len, record::EntrySource,
};

/// Encodes one sorted run of key/value entries into the on-disk block layout.
pub(crate) struct BlockBuilder<'a, S: EntrySource + ?Sized> {
    entries: &'a S,
    key_len: usize,
    value_layout: ValueLayout,
    block_checksum: BlockChecksumKind,
    value_payload_compression: ValuePayloadCompressionKind,
}

impl<'a, S: EntrySource + ?Sized> BlockBuilder<'a, S> {
    pub(crate) fn new(
        entries: &'a S,
        key_len: usize,
        value_layout: ValueLayout,
        block_checksum: BlockChecksumKind,
        value_payload_compression: ValuePayloadCompressionKind,
    ) -> Self {
        Self {
            entries,
            key_len,
            value_layout,
            block_checksum,
            value_payload_compression,
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
            .metadata_with_checksum_len(self.block_checksum)
            .map_err(|_| FormatError::limit("block payload offset"))?;
        let value_region = BlockValueRegion::for_write(
            self.value_layout,
            record_count,
            lookup_layout.key_section_len,
            payload_offset,
            payload_len,
        )
        .ok_or(FormatError::limit("block value region offsets"))?;
        let payload_frame_len = payload_len
            .checked_add(self.value_payload_compression.frame_header_len())
            .ok_or(FormatError::limit("block value payload frame length"))?;
        let capacity_len = value_region
            .payload_offset()
            .checked_add(payload_frame_len)
            .and_then(|len| len.checked_add(self.block_checksum.digest_len()))
            .ok_or(FormatError::limit("block length"))?;
        let mut block = Vec::with_capacity(capacity_len);
        self.write_keys(&mut block, key_prefix_len);
        value_region.write_index(self.entries, &mut block)?;
        debug_assert_eq!(block.len(), lookup_layout.lookup_metadata_len);
        append_checksum(
            self.block_checksum,
            &mut block,
            0..lookup_layout.lookup_metadata_len,
        );
        debug_assert_eq!(block.len(), value_region.payload_offset());
        let payload_start = block.len();
        match self.value_payload_compression {
            ValuePayloadCompressionKind::None => self.write_values(&mut block),
            #[cfg(feature = "value-compression-lz4")]
            ValuePayloadCompressionKind::Lz4 => {
                let mut raw_payload = Vec::with_capacity(payload_len);
                self.write_values(&mut raw_payload);
                let frame = self
                    .value_payload_compression
                    .encode_frame(&raw_payload, &mut block)?;
                debug_assert_eq!(frame.frame_len(), block.len() - payload_start);
            }
        }
        let payload_end = block.len();
        append_checksum(self.block_checksum, &mut block, payload_start..payload_end);
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

fn append_checksum(
    block_checksum: BlockChecksumKind,
    block: &mut Vec<u8>,
    checked_range: std::ops::Range<usize>,
) {
    let digest_len = block_checksum.digest_len();
    if digest_len == 0 {
        return;
    }
    let mut digest = [0u8; MAX_BLOCK_CHECKSUM_LEN];
    block_checksum.digest_into(&block[checked_range], &mut digest[..digest_len]);
    block.extend_from_slice(&digest[..digest_len]);
}
