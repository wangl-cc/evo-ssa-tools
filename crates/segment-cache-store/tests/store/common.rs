pub(crate) use std::{
    fs,
    fs::OpenOptions,
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

use crc32c::crc32c;
pub(crate) use segment_cache_store::{
    CommitStats, Error, Result, Store, StoreOptions, ValueLayout,
};

pub(crate) const FOOTER_MAGIC: &[u8; 8] = b"scsft001";

pub(crate) fn options(tempdir: &tempfile::TempDir) -> StoreOptions {
    StoreOptions::new(tempdir.path(), 16)
        .with_shard_count(4)
        .with_target_block_size(256)
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
    store.commit_batch(batch)
}

pub(crate) fn first_segment_path(root: &Path) -> Result<PathBuf> {
    let shards = root.join("shards");
    for shard in fs::read_dir(shards)? {
        let segment_dir = shard?.path().join("segments");
        if !segment_dir.exists() {
            continue;
        }
        let mut segments = fs::read_dir(segment_dir)?
            .map(|entry| entry.map(|entry| entry.path()))
            .collect::<std::io::Result<Vec<_>>>()?;
        segments.sort();
        if let Some(path) = segments
            .into_iter()
            .find(|path| path.extension().is_some_and(|extension| extension == "seg"))
        {
            return Ok(path);
        }
    }
    panic!("segment should exist");
}

pub(crate) fn block_offset(path: &Path, block_index: usize) -> Result<u64> {
    let mut file = OpenOptions::new().read(true).open(path)?;
    let mut offset = 0u64;
    for _ in 0..block_index {
        offset += u64::from(read_u32_at(&mut file, offset + 4)?);
    }
    Ok(offset)
}

pub(crate) fn block_index_offset(path: &Path) -> Result<u64> {
    let mut file = OpenOptions::new().read(true).open(path)?;
    let (_, footer) = read_footer_bytes(&mut file)?;
    Ok(u64::from_le_bytes(
        footer[32..40].try_into().expect("block index offset"),
    ))
}

pub(crate) fn mutate_footer_payload(path: &Path, mutate: impl FnOnce(&mut [u8])) -> Result<()> {
    let mut file = OpenOptions::new().read(true).write(true).open(path)?;
    let (footer_start, mut footer) = read_footer_bytes(&mut file)?;
    let payload_len = footer
        .len()
        .checked_sub(4)
        .expect("footer should include crc");
    mutate(&mut footer[..payload_len]);
    let crc = crc32c(&footer[..payload_len]);
    footer[payload_len..].copy_from_slice(&crc.to_le_bytes());
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
    let block_offset = block_offset(path, block_index)?;
    let mut file = OpenOptions::new().read(true).write(true).open(path)?;
    file.seek(SeekFrom::Start(block_offset + 4))?;
    let mut stored_len_bytes = [0u8; 4];
    file.read_exact(&mut stored_len_bytes)?;
    let stored_len = usize::try_from(u32::from_le_bytes(stored_len_bytes)).expect("block len");
    let mut block = vec![0u8; stored_len];
    file.seek(SeekFrom::Start(block_offset))?;
    file.read_exact(&mut block)?;

    let value_area_offset = usize::try_from(u32::from_le_bytes(
        block[8..12].try_into().expect("value area offset"),
    ))
    .expect("value area offset");
    let value_area_len = usize::try_from(u32::from_le_bytes(
        block[12..16].try_into().expect("value area len"),
    ))
    .expect("value area len");
    let payload_len = value_area_offset
        .checked_add(value_area_len)
        .ok_or(Error::CorruptBlock)?;
    if payload_len + 4 > block.len() {
        return Err(Error::CorruptBlock);
    }

    mutate(&mut block[..payload_len]);
    let crc = crc32c(&block[..payload_len]);
    block[payload_len..payload_len + 4].copy_from_slice(&crc.to_le_bytes());
    file.seek(SeekFrom::Start(block_offset))?;
    file.write_all(&block)?;
    file.sync_all()?;
    Ok(())
}

fn read_footer_bytes(file: &mut fs::File) -> Result<(u64, Vec<u8>)> {
    let trailer_len = 4u64 + u64::try_from(FOOTER_MAGIC.len()).expect("magic len");
    let file_len = file.metadata()?.len();
    file.seek(SeekFrom::End(
        -i64::try_from(trailer_len).expect("trailer len"),
    ))?;
    let mut trailer = [0u8; 12];
    file.read_exact(&mut trailer)?;
    assert_eq!(&trailer[4..], FOOTER_MAGIC);
    let footer_len = u32::from_le_bytes(trailer[..4].try_into().expect("footer len"));
    let footer_start = file_len - trailer_len - u64::from(footer_len);
    file.seek(SeekFrom::Start(footer_start))?;
    let mut footer = vec![0u8; usize::try_from(footer_len).expect("footer len")];
    file.read_exact(&mut footer)?;
    Ok((footer_start, footer))
}

fn read_u32_at(file: &mut fs::File, offset: u64) -> Result<u32> {
    file.seek(SeekFrom::Start(offset))?;
    let mut bytes = [0u8; 4];
    file.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}
