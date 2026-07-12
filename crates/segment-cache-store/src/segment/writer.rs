//! Deterministic encoding of complete immutable segment files.

use std::io::Write;

use super::{
    SegmentFingerprint, SegmentGeometry,
    fingerprint::FingerprintWriter,
    format::{SEGMENT_HEADER_LEN, SegmentFooter, SegmentHeader},
    index::{BlockIndexEntry, BlockKeyRange},
};
#[cfg(feature = "value-compression")]
use crate::block::{ValuePayloadCompressionPolicy, ValuePayloadEncoder};
use crate::{
    block::{BLOCK_METADATA_HEADER_LEN, BlockBuilder},
    error::{FormatError, SegmentWriteError},
    key::common_prefix_len,
    record::{EntrySource, EntryView},
};

/// Validated metadata produced while encoding one complete segment file.
pub(crate) struct SegmentFileMetadata {
    pub(super) footer: SegmentFooter,
    pub(super) min_key: Vec<u8>,
    pub(super) max_key: Vec<u8>,
    fingerprint: SegmentFingerprint,
}

impl SegmentFileMetadata {
    pub(crate) fn fingerprint(&self) -> SegmentFingerprint {
        self.fingerprint
    }
}

/// Deterministic encoder for one sorted immutable segment.
///
/// The same sorted entries written with the same parameters produce
/// byte-identical output, which keeps segment fingerprints and reproducibility
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
        let mut out = FingerprintWriter::new(out);
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

        let (record_count, block_index) = self.write_blocks(&mut out, entries)?;
        let (min_key, max_key) = if entries.is_empty() {
            (Vec::new(), Vec::new())
        } else {
            (entries.first_key().to_vec(), entries.last_key().to_vec())
        };
        let footer = SegmentFooter {
            record_count,
            block_index,
        };
        let mut footer_bytes = Vec::new();
        footer.write_to(self.geometry.key_len, &mut footer_bytes)?;
        out.write_all(&footer_bytes)?;
        let fingerprint = out.fingerprint();
        Ok(SegmentFileMetadata {
            footer,
            min_key,
            max_key,
            fingerprint,
        })
    }

    fn write_blocks<W: Write, S: EntrySource + ?Sized>(
        &self,
        out: &mut W,
        entries: &S,
    ) -> Result<(u64, Vec<BlockIndexEntry>), SegmentWriteError> {
        let mut block_index = Vec::new();
        let mut offset = SEGMENT_HEADER_LEN as u64;
        let mut record_count = 0u64;
        let mut start = 0usize;
        #[cfg(feature = "value-compression")]
        let mut payload_encoder = ValuePayloadEncoder::new(self.geometry.value_payload_compression);

        while start < entries.len() {
            let end = self.next_block_end(entries, start);
            let block_entries = EntryView::new(entries, start..end);
            let block_builder = self.block_builder(&block_entries);
            #[cfg(feature = "value-compression")]
            let block_bytes = block_builder.encode(&mut payload_encoder)?;
            #[cfg(not(feature = "value-compression"))]
            let block_bytes = block_builder.encode()?;
            let block_len =
                u64::try_from(block_bytes.len()).map_err(|_| FormatError::limit("block length"))?;
            let block_end = offset
                .checked_add(block_len)
                .ok_or(FormatError::limit("segment data length"))?;
            block_index.push(BlockIndexEntry {
                key_range: BlockKeyRange::new(block_entries.first_key(), block_entries.last_key())
                    .expect("sorted block entries define a valid key range"),
                byte_range: offset..block_end,
            });
            out.write_all(&block_bytes)?;
            offset = block_end;
            record_count += block_entries.len() as u64;
            start = end;
        }

        Ok((record_count, block_index))
    }

    #[cfg(feature = "value-compression")]
    fn block_builder<'a, S: EntrySource + ?Sized>(&self, entries: &'a S) -> BlockBuilder<'a, S> {
        BlockBuilder::new(
            entries,
            self.geometry.key_len,
            self.geometry.value_layout,
            self.geometry.block_checksum,
            self.geometry.value_payload_compression,
            self.value_payload_compression_policy,
        )
    }

    #[cfg(not(feature = "value-compression"))]
    fn block_builder<'a, S: EntrySource + ?Sized>(&self, entries: &'a S) -> BlockBuilder<'a, S> {
        BlockBuilder::new(
            entries,
            self.geometry.key_len,
            self.geometry.value_layout,
            self.geometry.block_checksum,
            self.geometry.value_payload_compression,
        )
    }

    fn next_block_end<S: EntrySource + ?Sized>(&self, entries: &S, start: usize) -> usize {
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
            let suffix_len = self.geometry.key_len - prefix_len;
            let key_region_len =
                BLOCK_METADATA_HEADER_LEN + prefix_len + prospective_count * suffix_len;
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
