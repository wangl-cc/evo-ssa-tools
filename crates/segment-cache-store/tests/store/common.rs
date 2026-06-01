pub(crate) use std::{
    fs,
    fs::OpenOptions,
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

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
    file.seek(SeekFrom::End(
        -4 - i64::try_from(FOOTER_MAGIC.len()).expect("magic len"),
    ))?;
    let mut trailer = [0u8; 12];
    file.read_exact(&mut trailer)?;
    assert_eq!(&trailer[4..], FOOTER_MAGIC);
    let footer_len = u32::from_le_bytes(trailer[..4].try_into().expect("footer len"));
    file.seek(SeekFrom::End(
        -i64::from(footer_len) - 4 - i64::try_from(FOOTER_MAGIC.len()).expect("magic len"),
    ))?;
    let mut footer = vec![0u8; usize::try_from(footer_len).expect("footer len")];
    file.read_exact(&mut footer)?;
    Ok(u64::from_le_bytes(
        footer[24..32].try_into().expect("block index offset"),
    ))
}

fn read_u32_at(file: &mut fs::File, offset: u64) -> Result<u32> {
    file.seek(SeekFrom::Start(offset))?;
    let mut bytes = [0u8; 4];
    file.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}
