use std::io::Write;

use super::{SEGMENT_HEADER_LEN, SegmentFooter, header::SegmentHeader, index::BlockIndexEntry};
use crate::format::{
    SegmentWriteError, ValueLayout,
    block::{BLOCK_FOOTER_LEN, BlockBuilder},
    common_prefix_len,
    entry::{EntrySource, EntryView},
    format_u32,
};

/// Deterministic encoder for one sorted immutable segment.
///
/// The same sorted entries written with the same parameters produce
/// byte-identical output; this determinism is load-bearing for
/// content-addressed identity and sync convergence (see `docs/design.md`).
pub(crate) struct SegmentWriter {
    key_len: usize,
    value_layout: ValueLayout,
    target_block_size: usize,
}

impl SegmentWriter {
    /// Creates a writer for one sorted immutable segment.
    pub(crate) fn new(key_len: usize, value_layout: ValueLayout, target_block_size: usize) -> Self {
        Self {
            key_len,
            value_layout,
            target_block_size,
        }
    }

    /// Encodes sorted entries plus the footer-owned block index into `out`.
    pub(crate) fn write<W: Write, S: EntrySource + ?Sized>(
        &self,
        out: &mut W,
        entries: &S,
    ) -> Result<(SegmentFooter, Vec<BlockIndexEntry>), SegmentWriteError> {
        let mut header = Vec::with_capacity(SEGMENT_HEADER_LEN);
        SegmentHeader::new(self.key_len, self.value_layout)?.into_bytes(&mut header);
        debug_assert_eq!(header.len(), SEGMENT_HEADER_LEN);
        out.write_all(&header)?;

        let (record_count, block_index) = self.write_blocks(out, entries)?;
        let footer = SegmentFooter {
            record_count,
            min_key: if entries.is_empty() {
                Vec::new()
            } else {
                entries.first_key().to_vec()
            },
            max_key: if entries.is_empty() {
                Vec::new()
            } else {
                entries.last_key().to_vec()
            },
            block_index,
        };
        let mut footer_bytes = Vec::new();
        footer.clone().into_bytes(self.key_len, &mut footer_bytes)?;
        out.write_all(&footer_bytes)?;
        Ok((footer.clone(), footer.block_index))
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

        while start < entries.len() {
            let end = self.next_block_end(entries, start);
            let block_entries = EntryView::new(entries, start..end);
            let block_bytes = BlockBuilder::new(
                &block_entries,
                self.key_len,
                self.value_layout,
                self.target_block_size,
            )
            .encode()?;
            let block_len = format_u32(block_bytes.len(), "block length")?;
            let block_record_count = format_u32(block_entries.len(), "block record count")?;
            block_index.push(BlockIndexEntry {
                first_key: block_entries.first_key().to_vec(),
                block_offset: offset,
                block_len,
                record_count: block_record_count,
            });
            out.write_all(&block_bytes)?;
            offset += u64::from(block_len);
            record_count += block_entries.len() as u64;
            start = end;
        }

        Ok((record_count, block_index))
    }

    fn next_block_end<S: EntrySource + ?Sized>(&self, entries: &S, start: usize) -> usize {
        let mut end = start;
        let mut payload_len = 0usize;
        let first_key = entries.entry(start).key();
        while end < entries.len() {
            let prospective_count = end - start + 1;
            let prefix_len = common_prefix_len(first_key, entries.entry(end).key());
            let suffix_len = self.key_len - prefix_len;
            let key_region_len = prefix_len + prospective_count * suffix_len;
            let value_index_len = if self.value_layout.is_variable() {
                (prospective_count + 1) * 4
            } else {
                0
            };
            let prospective_len = key_region_len
                + value_index_len
                + payload_len
                + entries.entry(end).value().len()
                + BLOCK_FOOTER_LEN;
            if end > start && prospective_len > self.target_block_size {
                break;
            }
            payload_len += entries.entry(end).value().len();
            end += 1;
        }
        end
    }
}
