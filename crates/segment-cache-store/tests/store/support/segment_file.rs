use std::{
    fs,
    fs::OpenOptions as FsOpenOptions,
    io::{Read, Seek, SeekFrom, Write},
    ops::Range,
    path::{Path, PathBuf},
};

use crc32c::crc32c;
use segment_cache_store::{BlockChecksumKind, CorruptionError, Result};

use crate::support::api::{block_checksum_digest_len, block_checksum_from_format_id};

pub(crate) const FOOTER_TRAILER_LEN: u64 = 8;
const KEY_PREFIX_LEN_LEN: usize = 4;
const VALUE_OFFSET_LEN: usize = 4;
const SEGMENT_VALUE_LEN_OFFSET: u64 = 12;
const SEGMENT_BLOCK_CHECKSUM_ID_OFFSET: u64 = 16;

pub(crate) fn first_segment_path(root: &Path) -> Result<PathBuf> {
    let mut segments = fs::read_dir(root.join("segments"))?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    segments.sort();
    if let Some(path) = segments
        .into_iter()
        .find(|path| path.extension().is_some_and(|extension| extension == "seg"))
    {
        return Ok(path);
    }
    panic!("segment should exist");
}

pub(crate) fn block_offset(path: &Path, block_index: usize) -> Result<u64> {
    Ok(block_index_entry(path, block_index)?.0)
}

pub(crate) fn block_index_offset(path: &Path) -> Result<u64> {
    let mut file = FsOpenOptions::new().read(true).open(path)?;
    let key_len = read_key_len(&mut file)?;
    let (footer_start, _) = read_footer_bytes(&mut file)?;
    Ok(footer_start + 8 + u64::try_from(key_len * 2).expect("key len fits") + 4)
}

pub(crate) fn mutate_footer_payload(path: &Path, mutate: impl FnOnce(&mut [u8])) -> Result<()> {
    let mut file = FsOpenOptions::new().read(true).write(true).open(path)?;
    let (footer_start, mut footer) = read_footer_bytes(&mut file)?;
    let payload_len = footer
        .len()
        .checked_sub(usize::try_from(FOOTER_TRAILER_LEN).expect("trailer len"))
        .expect("footer should include trailer");
    mutate(&mut footer[..payload_len]);
    let crc_offset = footer.len() - 4;
    let crc = crc32c(&footer[..crc_offset]);
    footer[crc_offset..].copy_from_slice(&crc.to_le_bytes());
    file.seek(SeekFrom::Start(footer_start))?;
    file.write_all(&footer)?;
    file.sync_all()?;
    Ok(())
}

pub(crate) fn truncate_first_block_to_declared_len(path: &Path, block_len: u32) -> Result<()> {
    let (block_offset, ..) = block_index_entry(path, 0)?;
    let mut file = FsOpenOptions::new().read(true).write(true).open(path)?;
    let key_len = read_key_len(&mut file)?;
    let (_, mut footer) = read_footer_bytes(&mut file)?;
    let payload_len = footer
        .len()
        .checked_sub(usize::try_from(FOOTER_TRAILER_LEN).expect("trailer len"))
        .expect("footer should include trailer");
    let block_count_offset = 8 + key_len * 2;
    let first_entry_offset = block_count_offset + 4;
    let block_len_offset = first_entry_offset + key_len + 8;
    assert!(block_len_offset + 4 <= payload_len);
    footer[block_len_offset..block_len_offset + 4].copy_from_slice(&block_len.to_le_bytes());
    let crc_offset = footer.len() - 4;
    let crc = crc32c(&footer[..crc_offset]);
    footer[crc_offset..].copy_from_slice(&crc.to_le_bytes());

    let footer_start = block_offset + u64::from(block_len);
    file.seek(SeekFrom::Start(footer_start))?;
    file.write_all(&footer)?;
    file.set_len(footer_start + u64::try_from(footer.len()).expect("footer len fits"))?;
    file.sync_all()?;
    Ok(())
}

pub(crate) fn mutate_block_metadata(
    path: &Path,
    block_index: usize,
    mutate: impl FnOnce(&mut [u8]),
) -> Result<()> {
    let (block_offset, block_len, record_count) = block_index_entry(path, block_index)?;
    let mut file = FsOpenOptions::new().read(true).write(true).open(path)?;
    let key_len = read_key_len(&mut file)?;
    let value_len = read_value_len(&mut file)?;
    let block_checksum = read_block_checksum(&mut file)?;
    let mut block = vec![0u8; usize::try_from(block_len).expect("block len")];
    file.seek(SeekFrom::Start(block_offset))?;
    file.read_exact(&mut block)?;

    let metadata_len =
        block_lookup_metadata_len(&block, record_count, key_len, value_len, block_checksum)?;
    mutate(&mut block[..metadata_len]);
    let checksum = block_checksum_digest(block_checksum, &block[..metadata_len]);
    block[metadata_len..metadata_len + checksum.len()].copy_from_slice(&checksum);
    file.seek(SeekFrom::Start(block_offset))?;
    file.write_all(&block)?;
    file.sync_all()?;
    Ok(())
}

pub(crate) fn corrupt_block_value_payload(path: &Path, block_index: usize) -> Result<()> {
    let (block_offset, block_len, record_count) = block_index_entry(path, block_index)?;
    let mut file = FsOpenOptions::new().read(true).write(true).open(path)?;
    let key_len = read_key_len(&mut file)?;
    let value_len = read_value_len(&mut file)?;
    let block_checksum = read_block_checksum(&mut file)?;
    let mut block = vec![0u8; usize::try_from(block_len).expect("block len")];
    file.seek(SeekFrom::Start(block_offset))?;
    file.read_exact(&mut block)?;
    let payload_range =
        block_value_payload_range(&block, record_count, key_len, value_len, block_checksum)?;
    let first_payload_byte = payload_range.start;
    block[first_payload_byte] ^= 0xFF;
    file.seek(SeekFrom::Start(block_offset))?;
    file.write_all(&block)?;
    file.sync_all()?;
    Ok(())
}

#[cfg(any(feature = "value-compression-lz4", feature = "value-compression-zstd"))]
pub(crate) fn corrupt_block_value_frame_start(path: &Path, block_index: usize) -> Result<()> {
    let (block_offset, block_len, record_count) = block_index_entry(path, block_index)?;
    let mut file = FsOpenOptions::new().read(true).write(true).open(path)?;
    let key_len = read_key_len(&mut file)?;
    let value_len = read_value_len(&mut file)?;
    let block_checksum = read_block_checksum(&mut file)?;
    let mut block = vec![0u8; usize::try_from(block_len).expect("block len")];
    file.seek(SeekFrom::Start(block_offset))?;
    file.read_exact(&mut block)?;
    let metadata_len =
        block_lookup_metadata_len(&block, record_count, key_len, value_len, block_checksum)?;
    let frame_offset = metadata_len
        .checked_add(block_checksum_digest_len(block_checksum))
        .ok_or(CorruptionError::Block)?;
    let byte = block.get_mut(frame_offset).ok_or(CorruptionError::Block)?;
    *byte ^= 0xFF;
    file.seek(SeekFrom::Start(block_offset))?;
    file.write_all(&block)?;
    file.sync_all()?;
    Ok(())
}

fn read_footer_bytes(file: &mut fs::File) -> Result<(u64, Vec<u8>)> {
    let file_len = file.metadata()?.len();
    file.seek(SeekFrom::End(
        -i64::try_from(FOOTER_TRAILER_LEN).expect("trailer len"),
    ))?;
    let mut trailer = [0u8; FOOTER_TRAILER_LEN as usize];
    file.read_exact(&mut trailer)?;
    let footer_len = u32::from_le_bytes(trailer[..4].try_into().expect("footer len"));
    let footer_start = file_len - FOOTER_TRAILER_LEN - u64::from(footer_len);
    file.seek(SeekFrom::Start(footer_start))?;
    let mut footer =
        vec![0u8; usize::try_from(u64::from(footer_len) + FOOTER_TRAILER_LEN).expect("footer len")];
    file.read_exact(&mut footer)?;
    Ok((footer_start, footer))
}

fn block_index_entry(path: &Path, block_index: usize) -> Result<(u64, u32, u32)> {
    let mut file = FsOpenOptions::new().read(true).open(path)?;
    let key_len = read_key_len(&mut file)?;
    let (_, footer) = read_footer_bytes(&mut file)?;
    let payload_len = footer.len() - usize::try_from(FOOTER_TRAILER_LEN).expect("trailer len");
    let payload = &footer[..payload_len];
    let mut cursor = 8 + key_len * 2;
    let block_count = usize::try_from(read_u32(payload, &mut cursor)?).expect("block count");
    assert!(block_index < block_count);
    for index in 0..block_count {
        cursor += key_len;
        let offset = read_u64(payload, &mut cursor)?;
        let len = read_u32(payload, &mut cursor)?;
        let record_count = read_u32(payload, &mut cursor)?;
        if index == block_index {
            return Ok((offset, len, record_count));
        }
    }
    unreachable!("block index checked above")
}

fn read_key_len(file: &mut fs::File) -> Result<usize> {
    Ok(usize::try_from(read_u32_at(file, 8)?).expect("key len"))
}

fn read_value_len(file: &mut fs::File) -> Result<u32> {
    read_u32_at(file, SEGMENT_VALUE_LEN_OFFSET)
}

fn read_block_checksum(file: &mut fs::File) -> Result<BlockChecksumKind> {
    let format_id = read_u32_at(file, SEGMENT_BLOCK_CHECKSUM_ID_OFFSET)?;
    block_checksum_from_format_id(format_id).ok_or(CorruptionError::SegmentFormat.into())
}

fn read_u32_at(file: &mut fs::File, offset: u64) -> Result<u32> {
    file.seek(SeekFrom::Start(offset))?;
    let mut bytes = [0u8; 4];
    file.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(bytes: &[u8], cursor: &mut usize) -> Result<u64> {
    if *cursor + 8 > bytes.len() {
        return Err(CorruptionError::Block.into());
    }
    let value = u64::from_le_bytes(bytes[*cursor..*cursor + 8].try_into().expect("u64"));
    *cursor += 8;
    Ok(value)
}

fn read_u32(bytes: &[u8], cursor: &mut usize) -> Result<u32> {
    if *cursor + 4 > bytes.len() {
        return Err(CorruptionError::Block.into());
    }
    let value = u32::from_le_bytes(bytes[*cursor..*cursor + 4].try_into().expect("u32"));
    *cursor += 4;
    Ok(value)
}

fn block_value_payload_range(
    block: &[u8],
    record_count: u32,
    key_len: usize,
    value_len: u32,
    block_checksum: BlockChecksumKind,
) -> Result<Range<usize>> {
    let metadata_len =
        block_lookup_metadata_len(block, record_count, key_len, value_len, block_checksum)?;
    let payload_offset = metadata_len
        .checked_add(block_checksum_digest_len(block_checksum))
        .ok_or(CorruptionError::Block)?;
    let payload_len =
        block_value_payload_len(block, record_count, key_len, value_len, block_checksum)?;
    let payload_end = payload_offset
        .checked_add(payload_len)
        .ok_or(CorruptionError::Block)?;
    if payload_end
        .checked_add(block_checksum_digest_len(block_checksum))
        .ok_or(CorruptionError::Block)?
        > block.len()
        || payload_offset > payload_end
    {
        return Err(CorruptionError::Block.into());
    }
    Ok(payload_offset..payload_end)
}

fn block_lookup_metadata_len(
    block: &[u8],
    record_count: u32,
    key_len: usize,
    value_len: u32,
    block_checksum: BlockChecksumKind,
) -> Result<usize> {
    if block.len() < KEY_PREFIX_LEN_LEN {
        return Err(CorruptionError::Block.into());
    }
    let prefix_len = u32::from_le_bytes(
        block[..KEY_PREFIX_LEN_LEN]
            .try_into()
            .expect("prefix length width"),
    ) as usize;
    if prefix_len > key_len {
        return Err(CorruptionError::Block.into());
    }
    let record_count = usize::try_from(record_count).expect("record count fits usize");
    let suffix_len = key_len - prefix_len;
    let key_section_len = KEY_PREFIX_LEN_LEN
        .checked_add(prefix_len)
        .and_then(|len| len.checked_add(record_count.checked_mul(suffix_len)?))
        .ok_or(CorruptionError::Block)?;
    let value_index_len = if value_len == 0 {
        record_count
            .checked_add(1)
            .and_then(|count| count.checked_mul(4))
            .ok_or(CorruptionError::Block)?
    } else {
        0
    };
    let metadata_len = key_section_len
        .checked_add(value_index_len)
        .ok_or(CorruptionError::Block)?;
    if metadata_len
        .checked_add(block_checksum_digest_len(block_checksum))
        .ok_or(CorruptionError::Block)?
        > block.len()
    {
        return Err(CorruptionError::Block.into());
    }
    Ok(metadata_len)
}

fn block_value_payload_len(
    block: &[u8],
    record_count: u32,
    key_len: usize,
    value_len: u32,
    block_checksum: BlockChecksumKind,
) -> Result<usize> {
    let record_count = usize::try_from(record_count).expect("record count fits usize");
    if value_len != 0 {
        return record_count
            .checked_mul(usize::try_from(value_len).expect("value len fits usize"))
            .ok_or_else(|| CorruptionError::Block.into());
    }
    let metadata_len = block_lookup_metadata_len(
        block,
        u32::try_from(record_count).expect("record count round-trips"),
        key_len,
        value_len,
        block_checksum,
    )?;
    let sentinel_offset = metadata_len
        .checked_sub(VALUE_OFFSET_LEN)
        .ok_or(CorruptionError::Block)?;
    let bytes = block
        .get(sentinel_offset..sentinel_offset + VALUE_OFFSET_LEN)
        .ok_or(CorruptionError::Block)?;
    Ok(u32::from_le_bytes(bytes.try_into().expect("value payload sentinel width")) as usize)
}

fn block_checksum_digest(checksum: BlockChecksumKind, bytes: &[u8]) -> Vec<u8> {
    #[cfg(not(any(feature = "checksum-crc32c", feature = "checksum-rapidhash")))]
    let _ = bytes;

    match checksum {
        BlockChecksumKind::None => Vec::new(),
        #[cfg(feature = "checksum-crc32c")]
        BlockChecksumKind::Crc32c => crc32c::crc32c(bytes).to_le_bytes().to_vec(),
        #[cfg(feature = "checksum-rapidhash")]
        BlockChecksumKind::RapidHashV3_64 => {
            rapidhash::v3::rapidhash_v3(bytes).to_le_bytes().to_vec()
        }
        _ => unreachable!("test fixture does not know this checksum kind"),
    }
}
