pub(crate) use std::{
    fs,
    fs::OpenOptions as FsOpenOptions,
    io::{Read, Seek, SeekFrom, Write},
    ops::Range,
    path::{Path, PathBuf},
};

use crc32c::crc32c;
pub(crate) use segment_cache_store::{
    BlockChecksumKind, CatalogError, CatalogMismatch, CommitOptions, CommitStats, CorruptionError,
    CreateOptions, Error, InputError, OpenOptions as StoreOpenOptions, OptionsError, Result, Store,
    StoreMetadata, ValueLayout,
};

pub(crate) const FOOTER_TRAILER_LEN: u64 = 8;
const KEY_PREFIX_LEN_LEN: usize = 4;
const VALUE_OFFSET_LEN: usize = 4;
const SEGMENT_VALUE_LEN_OFFSET: u64 = 12;
const SEGMENT_BLOCK_CHECKSUM_ID_OFFSET: u64 = 16;

pub(crate) fn metadata() -> StoreMetadata {
    StoreMetadata::from_text("segment-cache-store-test")
}

pub(crate) fn create_options() -> CreateOptions {
    CreateOptions::new(16, metadata())
}

pub(crate) fn open_options() -> StoreOpenOptions {
    StoreOpenOptions::new(metadata())
}

pub(crate) fn commit_options() -> CommitOptions {
    CommitOptions::default().with_target_block_size(256)
}

pub(crate) fn create_store(tempdir: &tempfile::TempDir) -> Result<Store> {
    Store::create(tempdir.path(), create_options())
}

pub(crate) fn create_store_with(
    tempdir: &tempfile::TempDir,
    options: CreateOptions,
) -> Result<Store> {
    Store::create(tempdir.path(), options)
}

pub(crate) fn reopen_store(tempdir: &tempfile::TempDir) -> Result<Store> {
    Store::open(tempdir.path(), open_options())
}

/// Opens a second read-only handle that takes no writer lock, so it can coexist
/// with a live writer handle to the same root.
pub(crate) fn reopen_store_read_only(tempdir: &tempfile::TempDir) -> Result<Store> {
    Store::open(tempdir.path(), open_options().with_read_only(true))
}

pub(crate) fn make_key(a: u32, b: u32, rep: u64) -> Vec<u8> {
    let mut key = Vec::with_capacity(16);
    key.extend_from_slice(&a.to_be_bytes());
    key.extend_from_slice(&b.to_be_bytes());
    key.extend_from_slice(&rep.to_be_bytes());
    key
}

pub(crate) fn make_value(tag: u8, len: usize) -> Vec<u8> {
    vec![tag; len]
}

pub(crate) fn commit_entries(
    store: &Store,
    entries: &[(Vec<u8>, Vec<u8>)],
    sorted: bool,
) -> Result<CommitStats> {
    let mut batch = store.begin_batch();
    for (key, value) in entries {
        batch.push(key, value)?;
    }
    if sorted {
        batch = batch.mark_sorted();
    }
    store.commit_batch_with_options(batch, &commit_options())
}

pub(crate) fn commit_entries_with_options(
    store: &Store,
    entries: &[(Vec<u8>, Vec<u8>)],
    sorted: bool,
    options: &CommitOptions,
) -> Result<CommitStats> {
    let mut batch = store.begin_batch();
    for (key, value) in entries {
        batch.push(key, value)?;
    }
    if sorted {
        batch = batch.mark_sorted();
    }
    store.commit_batch_with_options(batch, options)
}

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
    BlockChecksumKind::from_format_id(format_id).ok_or(CorruptionError::SegmentFormat.into())
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
        .checked_add(block_checksum.digest_len())
        .ok_or(CorruptionError::Block)?;
    let payload_len =
        block_value_payload_len(block, record_count, key_len, value_len, block_checksum)?;
    let payload_end = payload_offset
        .checked_add(payload_len)
        .ok_or(CorruptionError::Block)?;
    if payload_end
        .checked_add(block_checksum.digest_len())
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
        .checked_add(block_checksum.digest_len())
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
    let mut digest = vec![0u8; checksum.digest_len()];
    checksum.digest_into(bytes, &mut digest);
    digest
}
