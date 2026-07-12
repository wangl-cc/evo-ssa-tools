//! Block encoder: sorted entries in, on-disk block bytes out.

use super::{
    BlockChecksumKind, ValuePayloadCompressionKind,
    layout::{BlockLookupLayout, BlockValueRegion},
};
#[cfg(feature = "value-compression")]
use super::{ValuePayloadCompressionPolicy, ValuePayloadEncoder};
use crate::{
    binary::format_u32, error::FormatError, key::common_prefix_len, record::EntrySource,
    value::ValueLayout,
};

/// Encodes one sorted run of key/value entries into the on-disk block layout.
pub(crate) struct BlockBuilder<'a, S: EntrySource + ?Sized> {
    entries: &'a S,
    key_len: usize,
    value_layout: ValueLayout,
    block_checksum: BlockChecksumKind,
    #[cfg(feature = "value-compression")]
    value_payload_compression: ValuePayloadCompressionKind,
    #[cfg(feature = "value-compression")]
    value_payload_compression_policy: ValuePayloadCompressionPolicy,
}

impl<'a, S: EntrySource + ?Sized> BlockBuilder<'a, S> {
    #[cfg(feature = "value-compression")]
    pub(crate) fn new(
        entries: &'a S,
        key_len: usize,
        value_layout: ValueLayout,
        block_checksum: BlockChecksumKind,
        value_payload_compression: ValuePayloadCompressionKind,
        value_payload_compression_policy: ValuePayloadCompressionPolicy,
    ) -> Self {
        Self {
            entries,
            key_len,
            value_layout,
            block_checksum,
            value_payload_compression,
            value_payload_compression_policy,
        }
    }

    #[cfg(not(feature = "value-compression"))]
    pub(crate) fn new(
        entries: &'a S,
        key_len: usize,
        value_layout: ValueLayout,
        block_checksum: BlockChecksumKind,
        _value_payload_compression: ValuePayloadCompressionKind,
    ) -> Self {
        Self {
            entries,
            key_len,
            value_layout,
            block_checksum,
        }
    }

    pub(crate) fn encode(
        &self,
        #[cfg(feature = "value-compression")] payload_encoder: &mut ValuePayloadEncoder,
    ) -> Result<Vec<u8>, FormatError> {
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
        #[cfg(feature = "value-compression")]
        let payload_frame_len = payload_len
            .checked_add(self.value_payload_compression.frame_header_len())
            .ok_or(FormatError::limit("block value payload frame length"))?;
        #[cfg(not(feature = "value-compression"))]
        let payload_frame_len = payload_len;
        let capacity_len = value_region
            .payload_offset()
            .checked_add(payload_frame_len)
            .and_then(|len| len.checked_add(self.block_checksum.digest_len()))
            .ok_or(FormatError::limit("block length"))?;
        let mut block = Vec::with_capacity(capacity_len);
        block.extend_from_slice(&format_u32(record_count, "block record count")?.to_le_bytes());
        self.write_keys(&mut block, key_prefix_len);
        value_region.write_index(self.entries, &mut block)?;
        debug_assert_eq!(block.len(), lookup_layout.lookup_metadata_len);
        #[cfg(feature = "block-checksum")]
        {
            let digest = self
                .block_checksum
                .digest(&block[..lookup_layout.lookup_metadata_len]);
            block.extend_from_slice(digest.as_ref());
        }
        debug_assert_eq!(block.len(), value_region.payload_offset());
        #[cfg(feature = "block-checksum")]
        let payload_start = block.len();
        #[cfg(feature = "value-compression")]
        {
            let frame_start = block.len();
            let frame = payload_encoder.encode_frame(
                payload_len,
                self.value_payload_compression_policy,
                &mut block,
                |raw| self.write_values(raw),
            )?;
            debug_assert_eq!(frame.frame_len(), block.len() - frame_start);
        }
        #[cfg(not(feature = "value-compression"))]
        self.write_values(&mut block);
        #[cfg(feature = "block-checksum")]
        {
            let digest = self.block_checksum.digest(&block[payload_start..]);
            block.extend_from_slice(digest.as_ref());
        }
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
