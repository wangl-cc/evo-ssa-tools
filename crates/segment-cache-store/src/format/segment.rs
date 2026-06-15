//! Segment file format: header, footer, sparse block index, and writer.
//!
//! A segment is an immutable sorted file. The fixed-size header identifies
//! format and store geometry; the variable-size footer at the end of the file
//! is the completion marker and owns the sparse block index. Reading segment
//! bytes from disk lives in the engine; this module only encodes and decodes
//! them.

use std::io::Write;

use crc32c::crc32c;

use crate::format::{
    BlockChecksumKind, CorruptionError, FormatError, SegmentWriteError, ValueLayout,
    ValuePayloadCompressionKind, ValuePayloadCompressionPolicy, ValuePayloadEncoder,
    binary::BinaryCursor,
    block::{BlockBuilder, KEY_PREFIX_LEN_LEN},
    common_prefix_len, format_u32,
    record::{EntrySource, EntryView},
};

const SEGMENT_FORMAT_VERSION: u32 = 1;
pub(crate) const SEGMENT_HEADER_LEN: usize = 28;
pub(crate) const SEGMENT_FOOTER_TRAILER_LEN: usize = 8;
const HEADER_MAGIC: &[u8; 4] = b"SCSG";

// ─── SegmentHeader ────────────────────────────────────────────────────────────

/// Parsed fixed-size segment header.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SegmentHeader {
    version: u32,
    key_len: u32,
    value_layout: ValueLayout,
    block_checksum_id: u32,
    value_payload_compression_id: u32,
}

impl SegmentHeader {
    pub(crate) fn new(
        key_len: usize,
        value_layout: ValueLayout,
        block_checksum: BlockChecksumKind,
        value_payload_compression: ValuePayloadCompressionKind,
    ) -> Result<Self, FormatError> {
        Ok(Self {
            version: SEGMENT_FORMAT_VERSION,
            key_len: u32::try_from(key_len)
                .map_err(|_| FormatError::limit("segment key length"))?,
            value_layout,
            block_checksum_id: block_checksum.format_id(),
            value_payload_compression_id: value_payload_compression.format_id(),
        })
    }

    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self, CorruptionError> {
        if bytes.len() != SEGMENT_HEADER_LEN || &bytes[..4] != HEADER_MAGIC {
            return Err(CorruptionError::SegmentFormat);
        }
        let mut cursor = BinaryCursor::new(bytes);
        cursor.seek(24);
        let stored_crc = cursor.read::<u32>().ok_or(CorruptionError::SegmentFormat)?;
        if crc32c(&bytes[..24]) != stored_crc {
            return Err(CorruptionError::SegmentFormat);
        }
        cursor.seek(4);
        let version = cursor.read::<u32>().ok_or(CorruptionError::SegmentFormat)?;
        let key_len = cursor.read::<u32>().ok_or(CorruptionError::SegmentFormat)?;
        let value_len = cursor.read::<u32>().ok_or(CorruptionError::SegmentFormat)?;
        let block_checksum_id = cursor.read::<u32>().ok_or(CorruptionError::SegmentFormat)?;
        let value_payload_compression_id =
            cursor.read::<u32>().ok_or(CorruptionError::SegmentFormat)?;
        Ok(Self {
            version,
            key_len,
            value_layout: ValueLayout::from_u32(value_len),
            block_checksum_id,
            value_payload_compression_id,
        })
    }

    pub(crate) fn into_bytes(self, buffer: &mut Vec<u8>) {
        let start = buffer.len();
        buffer.extend_from_slice(HEADER_MAGIC);
        buffer.extend_from_slice(&self.version.to_le_bytes());
        buffer.extend_from_slice(&self.key_len.to_le_bytes());
        buffer.extend_from_slice(&self.value_layout.to_u32().to_le_bytes());
        buffer.extend_from_slice(&self.block_checksum_id.to_le_bytes());
        buffer.extend_from_slice(&self.value_payload_compression_id.to_le_bytes());
        let crc = crc32c(&buffer[start..start + 24]);
        buffer.extend_from_slice(&crc.to_le_bytes());
    }

    pub(crate) fn matches_geometry(
        self,
        expected_key_len: usize,
        expected_value_layout: ValueLayout,
        expected_block_checksum: BlockChecksumKind,
        expected_value_payload_compression: ValuePayloadCompressionKind,
    ) -> bool {
        self.version == SEGMENT_FORMAT_VERSION
            && self.key_len as usize == expected_key_len
            && self.value_layout == expected_value_layout
            && self.block_checksum_id == expected_block_checksum.format_id()
            && self.value_payload_compression_id == expected_value_payload_compression.format_id()
    }
}

// ─── SegmentFooter ────────────────────────────────────────────────────────────

/// Parsed segment footer.
///
/// The footer is the completion gate for a segment and owns the block index.
#[derive(Clone, Debug)]
pub(crate) struct SegmentFooter {
    pub(crate) record_count: u64,
    pub(crate) min_key: Vec<u8>,
    pub(crate) max_key: Vec<u8>,
    pub(crate) block_index: Vec<BlockIndexEntry>,
}

impl SegmentFooter {
    pub(crate) fn write_to(&self, key_len: usize, buffer: &mut Vec<u8>) -> Result<(), FormatError> {
        let footer_body_start = buffer.len();
        buffer.extend_from_slice(&self.record_count.to_le_bytes());
        buffer.extend_from_slice(&self.min_key);
        buffer.extend_from_slice(&self.max_key);
        BlockIndexCodec::new(key_len).write_to(&self.block_index, buffer)?;
        let footer_body_len = format_u32(
            buffer.len() - footer_body_start,
            "segment footer body length",
        )?;
        buffer.extend_from_slice(&footer_body_len.to_le_bytes());
        let crc = crc32c(&buffer[footer_body_start..]);
        buffer.extend_from_slice(&crc.to_le_bytes());
        Ok(())
    }

    /// Decodes a footer from the final `footer_body_len + 8` bytes of a
    /// segment file. `data_end` is the absolute file offset where the footer
    /// starts (the end of the data blocks), used to validate the block index.
    pub(crate) fn from_bytes(
        bytes: &[u8],
        key_len: usize,
        data_end: u64,
    ) -> Result<Self, CorruptionError> {
        if bytes.len() < SEGMENT_FOOTER_TRAILER_LEN {
            return Err(CorruptionError::SegmentFormat);
        }
        let footer_body_len_offset = bytes.len() - SEGMENT_FOOTER_TRAILER_LEN;
        let crc_offset = footer_body_len_offset + 4;
        let mut trailer_cursor = BinaryCursor::at(bytes, footer_body_len_offset);
        let footer_body_len = trailer_cursor
            .read::<u32>()
            .ok_or(CorruptionError::SegmentFormat)? as usize;
        if footer_body_len != footer_body_len_offset {
            return Err(CorruptionError::SegmentFormat);
        }
        trailer_cursor.seek(crc_offset);
        let stored_crc = trailer_cursor
            .read::<u32>()
            .ok_or(CorruptionError::SegmentFormat)?;
        if crc32c(&bytes[..crc_offset]) != stored_crc {
            return Err(CorruptionError::SegmentFormat);
        }

        let footer_body = &bytes[..footer_body_len];
        let mut cursor = BinaryCursor::new(footer_body);
        let record_count = cursor.read::<u64>().ok_or(CorruptionError::SegmentFormat)?;
        let min_key = cursor
            .read_vec(key_len)
            .ok_or(CorruptionError::SegmentFormat)?;
        let max_key = cursor
            .read_vec(key_len)
            .ok_or(CorruptionError::SegmentFormat)?;
        let block_index = BlockIndexCodec::new(key_len).read_entries(&mut cursor)?;
        let footer = Self {
            record_count,
            min_key,
            max_key,
            block_index,
        };
        BlockIndexCodec::new(key_len).validate_entries(&footer.block_index, &footer, data_end)?;
        Ok(footer)
    }
}

// ─── BlockIndexEntry + BlockIndexCodec ────────────────────────────────────────

/// Sparse index entry for one data block in a segment file.
#[derive(Clone, Debug)]
pub(crate) struct BlockIndexEntry {
    pub(crate) first_key: Vec<u8>,
    pub(crate) block_offset: u64,
    pub(crate) block_len: u32,
    pub(crate) record_count: u32,
}

struct BlockIndexCodec {
    key_len: usize,
}

impl BlockIndexCodec {
    fn new(key_len: usize) -> Self {
        Self { key_len }
    }

    fn write_to(
        &self,
        entries: &[BlockIndexEntry],
        buffer: &mut Vec<u8>,
    ) -> Result<(), FormatError> {
        buffer.extend_from_slice(
            &format_u32(entries.len(), "block index entry count")?.to_le_bytes(),
        );
        for entry in entries {
            buffer.extend_from_slice(&entry.first_key);
            buffer.extend_from_slice(&entry.block_offset.to_le_bytes());
            buffer.extend_from_slice(&entry.block_len.to_le_bytes());
            buffer.extend_from_slice(&entry.record_count.to_le_bytes());
        }
        Ok(())
    }

    fn read_entries(
        &self,
        cursor: &mut BinaryCursor<'_>,
    ) -> Result<Vec<BlockIndexEntry>, CorruptionError> {
        let count = cursor.read::<u32>().ok_or(CorruptionError::SegmentFormat)?;
        let count = count as usize;
        let expected_remaining = count
            .checked_mul(self.key_len + 8 + 4 + 4)
            .ok_or(CorruptionError::SegmentFormat)?;
        if cursor.remaining() != expected_remaining {
            return Err(CorruptionError::SegmentFormat);
        }
        let mut entries = Vec::with_capacity(count);
        for _ in 0..count {
            let first_key = cursor
                .read_vec(self.key_len)
                .ok_or(CorruptionError::SegmentFormat)?;
            let block_offset = cursor.read::<u64>().ok_or(CorruptionError::SegmentFormat)?;
            let block_len = cursor.read::<u32>().ok_or(CorruptionError::SegmentFormat)?;
            let record_count = cursor.read::<u32>().ok_or(CorruptionError::SegmentFormat)?;
            entries.push(BlockIndexEntry {
                first_key,
                block_offset,
                block_len,
                record_count,
            });
        }
        Ok(entries)
    }

    fn validate_entries(
        &self,
        entries: &[BlockIndexEntry],
        footer: &SegmentFooter,
        data_end: u64,
    ) -> Result<(), CorruptionError> {
        if entries.is_empty() || footer.record_count == 0 {
            return Err(CorruptionError::SegmentFormat);
        }
        if entries[0].first_key != footer.min_key {
            return Err(CorruptionError::SegmentFormat);
        }

        let mut expected_offset = SEGMENT_HEADER_LEN as u64;
        let mut decoded_records = 0u64;
        let mut previous_first_key: Option<&[u8]> = None;
        for entry in entries {
            if entry.block_len < KEY_PREFIX_LEN_LEN as u32
                || entry.record_count == 0
                || entry.block_offset != expected_offset
                || entry.first_key.len() != self.key_len
                || entry.first_key.as_slice() > footer.max_key.as_slice()
            {
                return Err(CorruptionError::SegmentFormat);
            }
            if let Some(previous_first_key) = previous_first_key
                && entry.first_key.as_slice() <= previous_first_key
            {
                return Err(CorruptionError::SegmentFormat);
            }
            expected_offset = expected_offset
                .checked_add(u64::from(entry.block_len))
                .ok_or(CorruptionError::SegmentFormat)?;
            decoded_records = decoded_records
                .checked_add(u64::from(entry.record_count))
                .ok_or(CorruptionError::SegmentFormat)?;
            previous_first_key = Some(entry.first_key.as_slice());
        }
        if expected_offset != data_end || decoded_records != footer.record_count {
            return Err(CorruptionError::SegmentFormat);
        }
        Ok(())
    }
}

// ─── SegmentWriter ────────────────────────────────────────────────────────────

/// Deterministic encoder for one sorted immutable segment.
///
/// The same sorted entries written with the same parameters produce
/// byte-identical output; this determinism is load-bearing for
/// content-addressed identity and sync convergence (see `docs/design.md`).
pub(crate) struct SegmentWriter {
    key_len: usize,
    value_layout: ValueLayout,
    block_checksum: BlockChecksumKind,
    value_payload_compression: ValuePayloadCompressionKind,
    value_payload_compression_policy: ValuePayloadCompressionPolicy,
    target_block_size: usize,
}

impl SegmentWriter {
    /// Creates a writer for one sorted immutable segment.
    pub(crate) fn new(
        key_len: usize,
        value_layout: ValueLayout,
        block_checksum: BlockChecksumKind,
        value_payload_compression: ValuePayloadCompressionKind,
        value_payload_compression_policy: ValuePayloadCompressionPolicy,
        target_block_size: usize,
    ) -> Self {
        Self {
            key_len,
            value_layout,
            block_checksum,
            value_payload_compression,
            value_payload_compression_policy,
            target_block_size,
        }
    }

    /// Encodes sorted entries plus the footer-owned block index into `out`.
    pub(crate) fn write<W: Write, S: EntrySource + ?Sized>(
        &self,
        out: &mut W,
        entries: &S,
    ) -> Result<SegmentFooter, SegmentWriteError> {
        let mut header = Vec::with_capacity(SEGMENT_HEADER_LEN);
        SegmentHeader::new(
            self.key_len,
            self.value_layout,
            self.block_checksum,
            self.value_payload_compression,
        )?
        .into_bytes(&mut header);
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
        footer.write_to(self.key_len, &mut footer_bytes)?;
        out.write_all(&footer_bytes)?;
        Ok(footer)
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
        let mut payload_encoder = ValuePayloadEncoder::new(self.value_payload_compression);

        while start < entries.len() {
            let end = self.next_block_end(entries, start);
            let block_entries = EntryView::new(entries, start..end);
            let block_bytes = BlockBuilder::new(
                &block_entries,
                self.key_len,
                self.value_layout,
                self.block_checksum,
                self.value_payload_compression,
                self.value_payload_compression_policy,
            )
            .encode(&mut payload_encoder)?;
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
            let entry = entries.entry(end);
            let prospective_count = end - start + 1;
            let prefix_len = common_prefix_len(first_key, entry.key());
            let suffix_len = self.key_len - prefix_len;
            let key_region_len = KEY_PREFIX_LEN_LEN + prefix_len + prospective_count * suffix_len;
            let value_index_len = if self.value_layout.is_variable() {
                (prospective_count + 1) * 4
            } else {
                0
            };
            let prospective_len = key_region_len
                + value_index_len
                + self.block_checksum.digest_len()
                + self.value_payload_compression.frame_header_len()
                + payload_len
                + entry.value().len()
                + self.block_checksum.digest_len();
            if end > start && prospective_len > self.target_block_size {
                break;
            }
            payload_len += entry.value().len();
            end += 1;
        }
        end
    }
}
