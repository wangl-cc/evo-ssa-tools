use std::{
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

use crc32c::crc32c;

pub(crate) use crate::block::{DecodedBlock, read_block, read_block_reusing};
use crate::{
    binary::{read_bytes, read_u32, read_u64},
    block::{BLOCK_HEADER_LEN, encode_block},
    error::{Error, Result},
    options::ValueLayout,
};

pub(crate) const SEGMENT_FORMAT_VERSION: u32 = 1;
pub(crate) const FOOTER_MAGIC: &[u8; 8] = b"scsft001";

#[derive(Clone, Debug)]
pub(crate) struct BlockIndexEntry {
    pub first_key: Vec<u8>,
    pub block_offset: u64,
    pub block_len: u32,
    pub record_count: u32,
}

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

#[derive(Debug)]
pub(crate) struct OpenedSegment {
    pub file: File,
    #[cfg(test)]
    pub path: PathBuf,
    pub min_key: Vec<u8>,
    pub max_key: Vec<u8>,
    pub block_index: Vec<BlockIndexEntry>,
}

pub(crate) fn shard_for_key(key: &[u8], shard_count: usize, shard_key_offset: usize) -> usize {
    if shard_count <= 1 {
        return 0;
    }

    let mut prefix = [0u8; 8];
    let shard_key = key.get(shard_key_offset..).unwrap_or_default();
    let copy_len = shard_key.len().min(prefix.len());
    prefix[..copy_len].copy_from_slice(&shard_key[..copy_len]);
    let position = u64::from_be_bytes(prefix);
    let shard_count = u128::try_from(shard_count).expect("shard count should fit in u128");
    let shard = ((u128::from(position) * shard_count) >> 64) as usize;
    shard.min(usize::try_from(shard_count - 1).expect("shard index should fit"))
}

pub(crate) fn write_segment(
    path: &Path,
    shard_id: usize,
    key_len: usize,
    value_layout: ValueLayout,
    codec_version: u32,
    target_block_size: usize,
    entries: &[(Vec<u8>, Vec<u8>)],
) -> Result<(SegmentFooter, Vec<BlockIndexEntry>)> {
    let mut file = File::create(path)?;
    let mut block_index = Vec::new();
    let mut offset = 0u64;
    let mut record_count = 0u64;
    let min_key = entries.first().map(|(k, _)| k.clone()).unwrap_or_default();
    let max_key = entries.last().map(|(k, _)| k.clone()).unwrap_or_default();

    let mut start = 0usize;
    while start < entries.len() {
        let mut end = start;
        let mut encoded_values_len = 0usize;
        while end < entries.len() {
            let prospective_count = end - start + 1;
            let keys_len = prospective_count * key_len;
            let value_table_len = match value_layout {
                ValueLayout::Variable => prospective_count * 8,
                ValueLayout::Fixed { .. } => 0,
            };
            let prospective_len = BLOCK_HEADER_LEN
                + keys_len
                + value_table_len
                + encoded_values_len
                + entries[end].1.len()
                + 4;
            if end > start && prospective_len > target_block_size {
                break;
            }
            encoded_values_len += entries[end].1.len();
            end += 1;
        }

        let block_entries = &entries[start..end];
        let block_bytes = encode_block(block_entries, key_len, value_layout, target_block_size);
        let block_len = u32::try_from(block_bytes.len()).expect("block length should fit in u32");
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

    let block_index_offset = offset;
    let block_index_bytes = encode_block_index(&block_index, key_len);
    let block_index_len =
        u64::try_from(block_index_bytes.len()).expect("block index length should fit");
    let block_index_crc = crc32c(&block_index_bytes);
    file.write_all(&block_index_bytes)?;

    let footer = SegmentFooter {
        version: SEGMENT_FORMAT_VERSION,
        shard_id: u32::try_from(shard_id).expect("shard id should fit in u32"),
        key_len: u32::try_from(key_len).expect("key length should fit in u32"),
        value_layout,
        codec_version,
        record_count,
        block_index_offset,
        block_index_len,
        block_index_crc,
        min_key,
        max_key,
    };
    let footer_bytes = encode_footer(&footer);
    file.write_all(&footer_bytes)?;
    file.sync_all()?;
    Ok((footer, block_index))
}

fn encode_block_index(entries: &[BlockIndexEntry], key_len: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(4 + entries.len() * (key_len + 8 + 4 + 4));
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

fn encode_footer(footer: &SegmentFooter) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&footer.version.to_le_bytes());
    payload.extend_from_slice(&footer.shard_id.to_le_bytes());
    payload.extend_from_slice(&footer.key_len.to_le_bytes());
    payload.extend_from_slice(&value_layout_tag(footer.value_layout).to_le_bytes());
    payload.extend_from_slice(&fixed_value_len(footer.value_layout).to_le_bytes());
    payload.extend_from_slice(&footer.codec_version.to_le_bytes());
    payload.extend_from_slice(&footer.record_count.to_le_bytes());
    payload.extend_from_slice(&footer.block_index_offset.to_le_bytes());
    payload.extend_from_slice(&footer.block_index_len.to_le_bytes());
    payload.extend_from_slice(&footer.block_index_crc.to_le_bytes());
    payload.extend_from_slice(&footer.min_key);
    payload.extend_from_slice(&footer.max_key);
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

pub(crate) fn open_segment(
    path: PathBuf,
    expected_shard: usize,
    expected_key_len: usize,
    expected_value_layout: ValueLayout,
    expected_codec_version: u32,
) -> Result<Option<OpenedSegment>> {
    let mut file = match File::open(&path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    let footer = match read_footer(&mut file, expected_key_len) {
        Ok(footer) => footer,
        Err(_) => return Ok(None),
    };
    if footer.version != SEGMENT_FORMAT_VERSION
        || usize::try_from(footer.shard_id).ok() != Some(expected_shard)
        || usize::try_from(footer.key_len).ok() != Some(expected_key_len)
        || footer.value_layout != expected_value_layout
        || footer.codec_version != expected_codec_version
    {
        return Ok(None);
    }
    let block_index = match read_block_index(&mut file, &footer, expected_key_len) {
        Ok(index) => index,
        Err(_) => return Ok(None),
    };
    Ok(Some(OpenedSegment {
        file,
        #[cfg(test)]
        path,
        min_key: footer.min_key,
        max_key: footer.max_key,
        block_index,
    }))
}

fn read_footer(file: &mut File, key_len: usize) -> Result<SegmentFooter> {
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
    let footer_payload_len = usize::try_from(footer_len).expect("footer len should fit");
    file.seek(SeekFrom::End(
        -i64::try_from(footer_total_len).expect("footer total len should fit"),
    ))?;
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
    Ok(SegmentFooter {
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

fn value_layout_tag(value_layout: ValueLayout) -> u32 {
    match value_layout {
        ValueLayout::Variable => 0,
        ValueLayout::Fixed { .. } => 1,
    }
}

fn fixed_value_len(value_layout: ValueLayout) -> u32 {
    match value_layout {
        ValueLayout::Variable => 0,
        ValueLayout::Fixed { value_len } => {
            u32::try_from(value_len).expect("fixed value length should fit in u32")
        }
    }
}

fn read_value_layout(bytes: &[u8], cursor: &mut usize) -> Result<ValueLayout> {
    let tag = read_u32(bytes, cursor)?;
    let fixed_len = read_u32(bytes, cursor)?;
    match (tag, fixed_len) {
        (0, 0) => Ok(ValueLayout::Variable),
        (1, len) if len > 0 => Ok(ValueLayout::Fixed {
            value_len: usize::try_from(len).expect("fixed value length should fit"),
        }),
        _ => Err(Error::UnsupportedFormatVersion { version: 0 }),
    }
}

fn read_block_index(
    file: &mut File,
    footer: &SegmentFooter,
    key_len: usize,
) -> Result<Vec<BlockIndexEntry>> {
    file.seek(SeekFrom::Start(footer.block_index_offset))?;
    let mut bytes =
        vec![0u8; usize::try_from(footer.block_index_len).expect("block index len should fit")];
    file.read_exact(&mut bytes)?;
    if crc32c(&bytes) != footer.block_index_crc {
        return Err(Error::UnsupportedFormatVersion {
            version: footer.version,
        });
    }
    let mut cursor = 0usize;
    let count = usize::try_from(read_u32(&bytes, &mut cursor)?).expect("count should fit");
    let mut entries = Vec::with_capacity(count);
    for _ in 0..count {
        let first_key = read_bytes(&bytes, &mut cursor, key_len)?;
        let block_offset = read_u64(&bytes, &mut cursor)?;
        let block_len = read_u32(&bytes, &mut cursor)?;
        let record_count = read_u32(&bytes, &mut cursor)?;
        entries.push(BlockIndexEntry {
            first_key,
            block_offset,
            block_len,
            record_count,
        });
    }
    Ok(entries)
}
