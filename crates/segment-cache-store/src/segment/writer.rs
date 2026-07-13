//! Deterministic encoding of complete immutable segment files.

use std::io::Write;

use super::{
    SegmentContentId, SegmentGeometry,
    content_id::ContentIdWriter,
    format::{SEGMENT_HEADER_LEN, SegmentFooter, SegmentHeader},
    index::{SegmentIndex, SegmentIndexBuilder},
};
#[cfg(feature = "value-compression")]
use crate::block::{ValuePayloadCompressionPolicy, ValuePayloadEncoder};
use crate::{
    block::{BLOCK_METADATA_HEADER_LEN, BlockBuilder},
    error::{FormatError, SegmentWriteError},
    key::common_prefix_len,
    limits::{MAX_BLOCK_COUNT, MAX_ENCODED_BLOCK_LEN},
    record::{EntrySource, EntryView},
};

/// Validated metadata produced while encoding one complete segment file.
pub(crate) struct SegmentFileMetadata {
    pub(super) footer: SegmentFooter,
    pub(super) min_key: Vec<u8>,
    pub(super) max_key: Vec<u8>,
    segment_len: u64,
    content_id: SegmentContentId,
}

impl SegmentFileMetadata {
    pub(crate) fn segment_len(&self) -> u64 {
        self.segment_len
    }

    pub(crate) fn content_id(&self) -> SegmentContentId {
        self.content_id
    }
}

/// Deterministic encoder for one sorted immutable segment.
///
/// The same sorted entries written with the same parameters produce
/// byte-identical output, which keeps content identities and reproducibility
/// tests stable.
pub(crate) struct SegmentWriter {
    geometry: SegmentGeometry,
    #[cfg(feature = "value-compression")]
    value_payload_compression_policy: ValuePayloadCompressionPolicy,
    target_block_size: usize,
}

impl SegmentWriter {
    /// Creates a writer for one sorted immutable segment.
    #[cfg(feature = "value-compression")]
    pub(crate) fn new(
        geometry: SegmentGeometry,
        value_payload_compression_policy: ValuePayloadCompressionPolicy,
        target_block_size: usize,
    ) -> Self {
        Self {
            geometry,
            value_payload_compression_policy,
            target_block_size,
        }
    }

    /// Creates a writer for one sorted immutable segment.
    #[cfg(not(feature = "value-compression"))]
    pub(crate) fn new(geometry: SegmentGeometry, target_block_size: usize) -> Self {
        Self {
            geometry,
            target_block_size,
        }
    }

    /// Encodes sorted entries plus the footer-owned block index into `out`.
    pub(crate) fn write<W: Write, S: EntrySource + ?Sized>(
        &self,
        out: &mut W,
        entries: &S,
    ) -> Result<SegmentFileMetadata, SegmentWriteError> {
        let mut out = ContentIdWriter::new(out);
        let mut header = Vec::with_capacity(SEGMENT_HEADER_LEN);
        SegmentHeader::new(
            self.geometry.key_len,
            self.geometry.value_layout,
            self.geometry.block_checksum,
            self.geometry.value_payload_compression,
        )?
        .into_bytes(&mut header);
        debug_assert_eq!(header.len(), SEGMENT_HEADER_LEN);
        out.write_all(&header)?;

        if entries.is_empty() {
            return Err(FormatError::limit("empty segment").into());
        }
        let min_key = entries.first_key().to_vec();
        let max_key = entries.last_key().to_vec();
        let block_index = self.write_blocks(&mut out, entries, &min_key, &max_key)?;
        let footer = SegmentFooter { block_index };
        let mut footer_bytes = Vec::new();
        footer.write_to(self.geometry.key_len, &mut footer_bytes)?;
        out.write_all(&footer_bytes)?;
        let segment_len = out.len();
        let content_id = out.content_id();
        Ok(SegmentFileMetadata {
            footer,
            min_key,
            max_key,
            segment_len,
            content_id,
        })
    }

    fn write_blocks<W: Write, S: EntrySource + ?Sized>(
        &self,
        out: &mut W,
        entries: &S,
        min_key: &[u8],
        max_key: &[u8],
    ) -> Result<SegmentIndex, SegmentWriteError> {
        let mut block_index = SegmentIndexBuilder::new(min_key, max_key)
            .ok_or_else(|| FormatError::limit("segment key range"))?;
        let segment_prefix_len = block_index.segment_prefix_len();
        let mut offset = SEGMENT_HEADER_LEN as u64;
        let mut start = 0usize;
        #[cfg(feature = "value-compression")]
        let mut payload_encoder = ValuePayloadEncoder::new(self.geometry.value_payload_compression);

        while start < entries.len() {
            if block_index.len() >= MAX_BLOCK_COUNT {
                return Err(FormatError::limit("segment block count").into());
            }
            let end = self.next_block_end(entries, start, segment_prefix_len);
            let block_entries = EntryView::new(entries, start..end);
            let block_builder = self.block_builder(&block_entries, segment_prefix_len);
            #[cfg(feature = "value-compression")]
            let block_bytes = block_builder.encode(&mut payload_encoder)?;
            #[cfg(not(feature = "value-compression"))]
            let block_bytes = block_builder.encode()?;
            let block_len =
                u64::try_from(block_bytes.len()).map_err(|_| FormatError::limit("block length"))?;
            if block_bytes.len() > MAX_ENCODED_BLOCK_LEN {
                return Err(FormatError::limit("block length").into());
            }
            let block_end = offset
                .checked_add(block_len)
                .ok_or(FormatError::limit("segment data length"))?;
            block_index
                .push(
                    block_entries.first_key(),
                    block_entries.last_key(),
                    offset..block_end,
                )
                .expect("sorted block entries share the segment prefix");
            out.write_all(&block_bytes)?;
            offset = block_end;
            start = end;
        }

        Ok(block_index.finish())
    }

    #[cfg(feature = "value-compression")]
    fn block_builder<'a, S: EntrySource + ?Sized>(
        &self,
        entries: &'a S,
        segment_prefix_len: usize,
    ) -> BlockBuilder<'a, S> {
        BlockBuilder::new(
            entries,
            self.geometry.key_len,
            segment_prefix_len,
            self.geometry.value_layout,
            self.geometry.block_checksum,
            self.geometry.value_payload_compression,
            self.value_payload_compression_policy,
        )
    }

    #[cfg(not(feature = "value-compression"))]
    fn block_builder<'a, S: EntrySource + ?Sized>(
        &self,
        entries: &'a S,
        segment_prefix_len: usize,
    ) -> BlockBuilder<'a, S> {
        BlockBuilder::new(
            entries,
            self.geometry.key_len,
            segment_prefix_len,
            self.geometry.value_layout,
            self.geometry.block_checksum,
            self.geometry.value_payload_compression,
        )
    }

    fn next_block_end<S: EntrySource + ?Sized>(
        &self,
        entries: &S,
        start: usize,
        segment_prefix_len: usize,
    ) -> usize {
        let mut end = start;
        let mut payload_len = 0usize;
        let first_key = entries.entry(start).key();
        #[cfg(feature = "value-compression")]
        let payload_frame_header_len = self.geometry.value_payload_compression.frame_header_len();
        #[cfg(not(feature = "value-compression"))]
        let payload_frame_header_len = 0;
        while end < entries.len() {
            let entry = entries.entry(end);
            let prospective_count = end - start + 1;
            let prefix_len = common_prefix_len(first_key, entry.key());
            let extra_prefix_len = prefix_len - segment_prefix_len;
            let suffix_len = self.geometry.key_len - segment_prefix_len - extra_prefix_len;
            let key_region_len =
                BLOCK_METADATA_HEADER_LEN + extra_prefix_len + prospective_count * suffix_len;
            let value_index_len = if self.geometry.value_layout.is_variable() {
                (prospective_count + 1) * 4
            } else {
                0
            };
            let prospective_len = key_region_len
                + value_index_len
                + self.geometry.block_checksum.digest_len()
                + payload_frame_header_len
                + payload_len
                + entry.value().len()
                + self.geometry.block_checksum.digest_len();
            if end > start && prospective_len > self.target_block_size {
                break;
            }
            payload_len += entry.value().len();
            end += 1;
        }
        end
    }
}
