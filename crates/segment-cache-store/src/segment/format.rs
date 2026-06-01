//! Segment file format.
//!
//! This module owns the stable on-disk segment contract: sharding policy,
//! segment writing, footer encoding, and block-index encoding. Data-block
//! internals live in `block.rs`; this module decides where those blocks live in
//! a segment file and how a segment is opened again.

use std::{
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

use crc32c::crc32c;

pub(crate) use crate::segment::block::{DecodedBlock, read_block, read_block_reusing};
use crate::{
    error::{Error, Result},
    io::binary::{read_bytes, read_u32, read_u64},
    options::ValueLayout,
    segment::block::{BLOCK_HEADER_LEN, BlockBuilder},
};

pub(crate) const SEGMENT_FORMAT_VERSION: u32 = 1;
pub(crate) const FOOTER_MAGIC: &[u8; 8] = b"scsft001";

/// Lexicographic-prefix shard assignment persisted by manifest options.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ShardPolicy {
    count: usize,
    key_offset: usize,
}

impl ShardPolicy {
    pub(crate) fn new(count: usize, key_offset: usize) -> Self {
        Self { count, key_offset }
    }

    pub(crate) fn shard_for_key(self, key: &[u8]) -> usize {
        if self.count <= 1 {
            return 0;
        }

        let mut prefix = [0u8; 8];
        let shard_key = key.get(self.key_offset..).unwrap_or_default();
        let copy_len = shard_key.len().min(prefix.len());
        prefix[..copy_len].copy_from_slice(&shard_key[..copy_len]);
        let position = u64::from_be_bytes(prefix);
        let shard_count = u128::try_from(self.count).expect("shard count should fit in u128");
        let shard = ((u128::from(position) * shard_count) >> 64) as usize;
        shard.min(usize::try_from(shard_count - 1).expect("shard index should fit"))
    }
}

/// Sparse index entry for one data block in a segment file.
#[derive(Clone, Debug)]
pub(crate) struct BlockIndexEntry {
    pub first_key: Vec<u8>,
    pub block_offset: u64,
    pub block_len: u32,
    pub record_count: u32,
}

struct BlockIndexCodec {
    key_len: usize,
}

impl BlockIndexCodec {
    fn new(key_len: usize) -> Self {
        Self { key_len }
    }

    fn encode(&self, entries: &[BlockIndexEntry]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(4 + entries.len() * (self.key_len + 8 + 4 + 4));
        bytes.extend_from_slice(
            &u32::try_from(entries.len())
                .expect("block index entry count should fit in u32")
                .to_le_bytes(),
        );
        for entry in entries {
            bytes.extend_from_slice(&entry.first_key);
            bytes.extend_from_slice(&entry.block_offset.to_le_bytes());
            bytes.extend_from_slice(&entry.block_len.to_le_bytes());
            bytes.extend_from_slice(&entry.record_count.to_le_bytes());
        }
        bytes
    }

    fn read_from(&self, file: &mut File, footer: &SegmentFooter) -> Result<Vec<BlockIndexEntry>> {
        let file_len = file.metadata()?.len();
        self.validate_index_region(footer, file_len)?;
        file.seek(SeekFrom::Start(footer.block_index_offset))?;
        let mut count_bytes = [0u8; 4];
        file.read_exact(&mut count_bytes)?;
        let declared_count = u32::from_le_bytes(count_bytes);
        let expected_len = self.validate_declared_len(footer, declared_count)?;
        file.seek(SeekFrom::Start(footer.block_index_offset))?;
        let mut bytes = vec![0u8; expected_len];
        file.read_exact(&mut bytes)?;
        if crc32c(&bytes) != footer.block_index_crc {
            return Err(Error::UnsupportedFormatVersion {
                version: footer.version,
            });
        }
        let entries = self.decode_entries(&bytes)?;
        self.validate_entries(&entries, footer)?;
        Ok(entries)
    }

    fn validate_index_region(&self, footer: &SegmentFooter, file_len: u64) -> Result<()> {
        if footer.block_index_len < 4 || footer.block_index_offset > file_len {
            return Err(Error::UnsupportedFormatVersion {
                version: footer.version,
            });
        }
        let end = footer
            .block_index_offset
            .checked_add(footer.block_index_len)
            .ok_or(Error::UnsupportedFormatVersion {
                version: footer.version,
            })?;
        if end > file_len {
            return Err(Error::UnsupportedFormatVersion {
                version: footer.version,
            });
        }
        Ok(())
    }

    fn validate_declared_len(&self, footer: &SegmentFooter, declared_count: u32) -> Result<usize> {
        let declared_count = u64::from(declared_count);
        let per_entry_len = u64::try_from(self.key_len)
            .map_err(|_| Error::UnsupportedFormatVersion {
                version: footer.version,
            })?
            .checked_add(8 + 4 + 4)
            .ok_or(Error::UnsupportedFormatVersion {
                version: footer.version,
            })?;
        let expected_len = 4u64
            .checked_add(declared_count.checked_mul(per_entry_len).ok_or(
                Error::UnsupportedFormatVersion {
                    version: footer.version,
                },
            )?)
            .ok_or(Error::UnsupportedFormatVersion {
                version: footer.version,
            })?;
        if declared_count == 0
            || declared_count > footer.record_count
            || expected_len != footer.block_index_len
        {
            return Err(Error::UnsupportedFormatVersion {
                version: footer.version,
            });
        }
        usize::try_from(expected_len).map_err(|_| Error::UnsupportedFormatVersion {
            version: footer.version,
        })
    }

    fn decode_entries(&self, bytes: &[u8]) -> Result<Vec<BlockIndexEntry>> {
        let mut cursor = 0usize;
        let count = usize::try_from(read_u32(bytes, &mut cursor)?).expect("count should fit");
        let expected_len = 4usize
            .checked_add(
                count
                    .checked_mul(self.key_len + 8 + 4 + 4)
                    .ok_or(Error::UnsupportedFormatVersion { version: 0 })?,
            )
            .ok_or(Error::UnsupportedFormatVersion { version: 0 })?;
        if expected_len != bytes.len() {
            return Err(Error::UnsupportedFormatVersion { version: 0 });
        }
        let mut entries = Vec::with_capacity(count);
        for _ in 0..count {
            entries.push(BlockIndexEntry {
                first_key: read_bytes(bytes, &mut cursor, self.key_len)?,
                block_offset: read_u64(bytes, &mut cursor)?,
                block_len: read_u32(bytes, &mut cursor)?,
                record_count: read_u32(bytes, &mut cursor)?,
            });
        }
        Ok(entries)
    }

    fn validate_entries(&self, entries: &[BlockIndexEntry], footer: &SegmentFooter) -> Result<()> {
        if entries.is_empty() || footer.record_count == 0 {
            return Err(Error::UnsupportedFormatVersion {
                version: footer.version,
            });
        }
        if entries[0].first_key != footer.min_key {
            return Err(Error::UnsupportedFormatVersion {
                version: footer.version,
            });
        }

        let mut expected_offset = 0u64;
        let mut decoded_records = 0u64;
        let mut previous_first_key: Option<&[u8]> = None;
        for entry in entries {
            if entry.block_len == 0
                || entry.record_count == 0
                || entry.block_offset != expected_offset
                || entry.first_key.len() != self.key_len
                || entry.first_key.as_slice() > footer.max_key.as_slice()
            {
                return Err(Error::UnsupportedFormatVersion {
                    version: footer.version,
                });
            }
            if let Some(previous_first_key) = previous_first_key
                && entry.first_key.as_slice() <= previous_first_key
            {
                return Err(Error::UnsupportedFormatVersion {
                    version: footer.version,
                });
            }
            expected_offset = expected_offset
                .checked_add(u64::from(entry.block_len))
                .ok_or(Error::UnsupportedFormatVersion {
                    version: footer.version,
                })?;
            decoded_records = decoded_records
                .checked_add(u64::from(entry.record_count))
                .ok_or(Error::UnsupportedFormatVersion {
                    version: footer.version,
                })?;
            previous_first_key = Some(entry.first_key.as_slice());
        }
        if expected_offset != footer.block_index_offset || decoded_records != footer.record_count {
            return Err(Error::UnsupportedFormatVersion {
                version: footer.version,
            });
        }
        Ok(())
    }
}

/// Parsed segment footer.
///
/// The footer is the compatibility gate for a segment: if any field disagrees
/// with the store options, the whole segment is ignored.
#[derive(Clone, Debug)]
pub(crate) struct SegmentFooter {
    pub version: u32,
    pub shard_id: u32,
    pub key_len: u32,
    pub value_layout: ValueLayout,
    pub codec_version: u32,
    pub record_count: u64,
    pub block_index_offset: u64,
    pub block_index_len: u64,
    pub block_index_crc: u32,
    pub min_key: Vec<u8>,
    pub max_key: Vec<u8>,
}

impl SegmentFooter {
    fn encode(&self) -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend_from_slice(&self.version.to_le_bytes());
        payload.extend_from_slice(&self.shard_id.to_le_bytes());
        payload.extend_from_slice(&self.key_len.to_le_bytes());
        payload.extend_from_slice(&self.value_layout.segment_tag().to_le_bytes());
        payload.extend_from_slice(&self.value_layout.segment_fixed_len().to_le_bytes());
        payload.extend_from_slice(&self.codec_version.to_le_bytes());
        payload.extend_from_slice(&self.record_count.to_le_bytes());
        payload.extend_from_slice(&self.block_index_offset.to_le_bytes());
        payload.extend_from_slice(&self.block_index_len.to_le_bytes());
        payload.extend_from_slice(&self.block_index_crc.to_le_bytes());
        payload.extend_from_slice(&self.min_key);
        payload.extend_from_slice(&self.max_key);

        let footer_crc = crc32c(&payload);
        let mut bytes = payload;
        bytes.extend_from_slice(&footer_crc.to_le_bytes());
        bytes.extend_from_slice(
            &u32::try_from(bytes.len())
                .expect("footer length should fit in u32")
                .to_le_bytes(),
        );
        bytes.extend_from_slice(FOOTER_MAGIC);
        bytes
    }

    fn read_from(file: &mut File, key_len: usize) -> Result<Self> {
        let file_len = file.metadata()?.len();
        let trailer_len = 4u64 + u64::try_from(FOOTER_MAGIC.len()).expect("magic len should fit");
        if file_len < trailer_len {
            return Err(Error::UnsupportedFormatVersion { version: 0 });
        }
        file.seek(SeekFrom::End(
            -i64::try_from(trailer_len).expect("trailer len should fit"),
        ))?;
        let mut trailer = [0u8; 12];
        file.read_exact(&mut trailer)?;
        let footer_len = u32::from_le_bytes(trailer[..4].try_into().expect("footer len slice"));
        if &trailer[4..] != FOOTER_MAGIC {
            return Err(Error::UnsupportedFormatVersion { version: 0 });
        }

        let footer_total_len = u64::from(footer_len) + trailer_len;
        if file_len < footer_total_len {
            return Err(Error::UnsupportedFormatVersion { version: 0 });
        }
        file.seek(SeekFrom::End(
            -i64::try_from(footer_total_len).expect("footer total len should fit"),
        ))?;
        Self::decode_payload(file, footer_len, key_len)
    }

    fn decode_payload(file: &mut File, footer_len: u32, key_len: usize) -> Result<Self> {
        let footer_payload_len = usize::try_from(footer_len).expect("footer len should fit");
        let mut footer_bytes = vec![0u8; footer_payload_len];
        file.read_exact(&mut footer_bytes)?;
        if footer_bytes.len() < 4 + 4 + 4 + 4 + 4 + 4 + 8 + 8 + 8 + 4 + key_len * 2 + 4 {
            return Err(Error::UnsupportedFormatVersion { version: 0 });
        }
        let stored_crc = u32::from_le_bytes(
            footer_bytes[footer_bytes.len() - 4..]
                .try_into()
                .expect("footer crc slice"),
        );
        let payload = &footer_bytes[..footer_bytes.len() - 4];
        if crc32c(payload) != stored_crc {
            return Err(Error::UnsupportedFormatVersion { version: 0 });
        }

        let mut cursor = 0usize;
        let version = read_u32(payload, &mut cursor)?;
        let shard_id = read_u32(payload, &mut cursor)?;
        let stored_key_len = read_u32(payload, &mut cursor)?;
        let value_layout = read_value_layout(payload, &mut cursor)?;
        let codec_version = read_u32(payload, &mut cursor)?;
        let record_count = read_u64(payload, &mut cursor)?;
        let block_index_offset = read_u64(payload, &mut cursor)?;
        let block_index_len = read_u64(payload, &mut cursor)?;
        let block_index_crc = read_u32(payload, &mut cursor)?;
        let min_key = read_bytes(payload, &mut cursor, key_len)?;
        let max_key = read_bytes(payload, &mut cursor, key_len)?;
        if usize::try_from(stored_key_len).ok() != Some(key_len) {
            return Err(Error::UnsupportedFormatVersion { version });
        }

        Ok(Self {
            version,
            shard_id,
            key_len: stored_key_len,
            value_layout,
            codec_version,
            record_count,
            block_index_offset,
            block_index_len,
            block_index_crc,
            min_key,
            max_key,
        })
    }
}

fn read_value_layout(bytes: &[u8], cursor: &mut usize) -> Result<ValueLayout> {
    let tag = read_u32(bytes, cursor)?;
    let fixed_len = read_u32(bytes, cursor)?;
    ValueLayout::from_segment_fields(tag, fixed_len)
}

pub(crate) struct SegmentWriter<'a> {
    path: &'a Path,
    shard_id: usize,
    key_len: usize,
    value_layout: ValueLayout,
    codec_version: u32,
    target_block_size: usize,
}

impl<'a> SegmentWriter<'a> {
    /// Creates a writer for one unpublished temporary segment path.
    pub(crate) fn new(
        path: &'a Path,
        shard_id: usize,
        key_len: usize,
        value_layout: ValueLayout,
        codec_version: u32,
        target_block_size: usize,
    ) -> Self {
        Self {
            path,
            shard_id,
            key_len,
            value_layout,
            codec_version,
            target_block_size,
        }
    }

    /// Writes sorted entries, block index, and footer, then fsyncs the file.
    pub(crate) fn write(
        &self,
        entries: &[(Vec<u8>, Vec<u8>)],
    ) -> Result<(SegmentFooter, Vec<BlockIndexEntry>)> {
        let mut file = File::create(self.path)?;
        let (block_index_offset, record_count, block_index) =
            self.write_blocks(&mut file, entries)?;
        let block_index_bytes = BlockIndexCodec::new(self.key_len).encode(&block_index);
        let block_index_len =
            u64::try_from(block_index_bytes.len()).expect("block index length should fit");
        let block_index_crc = crc32c(&block_index_bytes);
        file.write_all(&block_index_bytes)?;

        let footer = SegmentFooter {
            version: SEGMENT_FORMAT_VERSION,
            shard_id: u32::try_from(self.shard_id).expect("shard id should fit in u32"),
            key_len: u32::try_from(self.key_len).expect("key length should fit in u32"),
            value_layout: self.value_layout,
            codec_version: self.codec_version,
            record_count,
            block_index_offset,
            block_index_len,
            block_index_crc,
            min_key: entries.first().map(|(k, _)| k.clone()).unwrap_or_default(),
            max_key: entries.last().map(|(k, _)| k.clone()).unwrap_or_default(),
        };
        file.write_all(&footer.encode())?;
        file.sync_all()?;
        Ok((footer, block_index))
    }

    fn write_blocks(
        &self,
        file: &mut File,
        entries: &[(Vec<u8>, Vec<u8>)],
    ) -> Result<(u64, u64, Vec<BlockIndexEntry>)> {
        let mut block_index = Vec::new();
        let mut offset = 0u64;
        let mut record_count = 0u64;
        let mut start = 0usize;

        while start < entries.len() {
            let end = self.next_block_end(entries, start);
            let block_entries = &entries[start..end];
            let block_bytes = BlockBuilder::new(
                block_entries,
                self.key_len,
                self.value_layout,
                self.target_block_size,
            )
            .encode();
            let block_len =
                u32::try_from(block_bytes.len()).expect("block length should fit in u32");
            block_index.push(BlockIndexEntry {
                first_key: block_entries[0].0.clone(),
                block_offset: offset,
                block_len,
                record_count: u32::try_from(block_entries.len())
                    .expect("record count should fit in u32"),
            });
            file.write_all(&block_bytes)?;
            offset += u64::from(block_len);
            record_count += u64::try_from(block_entries.len()).expect("record count should fit");
            start = end;
        }

        Ok((offset, record_count, block_index))
    }

    fn next_block_end(&self, entries: &[(Vec<u8>, Vec<u8>)], start: usize) -> usize {
        let mut end = start;
        let mut encoded_values_len = 0usize;
        while end < entries.len() {
            let prospective_count = end - start + 1;
            let keys_len = prospective_count * self.key_len;
            let value_table_len = match self.value_layout {
                ValueLayout::Variable => prospective_count * 8,
                ValueLayout::Fixed { .. } => 0,
            };
            let prospective_len = BLOCK_HEADER_LEN
                + keys_len
                + value_table_len
                + encoded_values_len
                + entries[end].1.len()
                + 4;
            if end > start && prospective_len > self.target_block_size {
                break;
            }
            encoded_values_len += entries[end].1.len();
            end += 1;
        }
        end
    }
}

#[derive(Clone, Copy)]
pub(crate) struct SegmentOpenOptions {
    pub expected_shard: usize,
    pub expected_key_len: usize,
    pub expected_value_layout: ValueLayout,
    pub expected_codec_version: u32,
}

/// Open segment handle with its sparse block index loaded into memory.
#[derive(Debug)]
pub(crate) struct OpenedSegment {
    pub file: File,
    pub min_key: Vec<u8>,
    pub max_key: Vec<u8>,
    pub record_count: u64,
    pub block_index: Vec<BlockIndexEntry>,
}

impl OpenedSegment {
    pub(crate) fn open(path: PathBuf, options: SegmentOpenOptions) -> Result<Option<Self>> {
        let mut file = match File::open(&path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let footer = match SegmentFooter::read_from(&mut file, options.expected_key_len) {
            Ok(footer) => footer,
            Err(_) => return Ok(None),
        };
        if !footer.matches_options(options) {
            return Ok(None);
        }
        let block_index =
            match BlockIndexCodec::new(options.expected_key_len).read_from(&mut file, &footer) {
                Ok(index) => index,
                Err(_) => return Ok(None),
            };
        Ok(Some(Self {
            file,
            min_key: footer.min_key,
            max_key: footer.max_key,
            record_count: footer.record_count,
            block_index,
        }))
    }
}

impl SegmentFooter {
    fn matches_options(&self, options: SegmentOpenOptions) -> bool {
        self.version == SEGMENT_FORMAT_VERSION
            && usize::try_from(self.shard_id).ok() == Some(options.expected_shard)
            && usize::try_from(self.key_len).ok() == Some(options.expected_key_len)
            && self.value_layout == options.expected_value_layout
            && self.codec_version == options.expected_codec_version
    }
}
