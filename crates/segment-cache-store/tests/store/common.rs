pub(crate) use std::{
    fs,
    fs::OpenOptions as FsOpenOptions,
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

use crc32c::crc32c;
pub(crate) use segment_cache_store::{
    CatalogError, CatalogMismatch, CommitOptions, CommitStats, CorruptionError, CreateOptions,
    Error, InputError, OpenOptions as StoreOpenOptions, OptionsError, Result, Store, StoreMetadata,
    ValueLayout,
};

pub(crate) const SEGMENT_HEADER_LEN: u64 = 24;
pub(crate) const FOOTER_TRAILER_LEN: u64 = 8;
const BLOCK_FOOTER_LEN: usize = 16;

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

pub(crate) fn mutate_block_payload(
    path: &Path,
    block_index: usize,
    mutate: impl FnOnce(&mut [u8]),
) -> Result<()> {
    let (block_offset, block_len, _) = block_index_entry(path, block_index)?;
    let mut file = FsOpenOptions::new().read(true).write(true).open(path)?;
    let mut block = vec![0u8; usize::try_from(block_len).expect("block len")];
    file.seek(SeekFrom::Start(block_offset))?;
    file.read_exact(&mut block)?;

    if block.len() < BLOCK_FOOTER_LEN {
        return Err(CorruptionError::Block.into());
    }
    let crc_offset = block.len() - 4;

    mutate(&mut block[..crc_offset]);
    let crc = crc32c(&block[..crc_offset]);
    block[crc_offset..].copy_from_slice(&crc.to_le_bytes());
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
