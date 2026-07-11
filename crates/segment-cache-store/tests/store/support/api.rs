use segment_cache_store::{
    BlockChecksumKind, CommitOptions, CommitStats, CreateOptions, OpenOptions as StoreOpenOptions,
    Result, Store, StoreMetadata, WriteBatch,
};

pub(crate) fn metadata() -> StoreMetadata {
    StoreMetadata::from_text("segment-cache-store-test")
}

pub(crate) fn test_block_checksum() -> BlockChecksumKind {
    #[cfg(feature = "checksum-rapidhash")]
    {
        BlockChecksumKind::RapidHashV3_64
    }
    #[cfg(all(not(feature = "checksum-rapidhash"), feature = "checksum-crc32c"))]
    {
        BlockChecksumKind::Crc32c
    }
    #[cfg(not(any(feature = "checksum-rapidhash", feature = "checksum-crc32c")))]
    {
        BlockChecksumKind::None
    }
}

pub(crate) fn block_checksum_format_id(checksum: BlockChecksumKind) -> u32 {
    match checksum {
        BlockChecksumKind::None => 0,
        #[cfg(feature = "checksum-crc32c")]
        BlockChecksumKind::Crc32c => 1,
        #[cfg(feature = "checksum-rapidhash")]
        BlockChecksumKind::RapidHashV3_64 => 2,
        _ => unreachable!("test fixture does not know this checksum kind"),
    }
}

pub(crate) fn block_checksum_digest_len(checksum: BlockChecksumKind) -> usize {
    match checksum {
        BlockChecksumKind::None => 0,
        #[cfg(feature = "checksum-crc32c")]
        BlockChecksumKind::Crc32c => 4,
        #[cfg(feature = "checksum-rapidhash")]
        BlockChecksumKind::RapidHashV3_64 => 8,
        _ => unreachable!("test fixture does not know this checksum kind"),
    }
}

pub(crate) fn block_checksum_from_format_id(format_id: u32) -> Option<BlockChecksumKind> {
    match format_id {
        0 => Some(BlockChecksumKind::None),
        #[cfg(feature = "checksum-crc32c")]
        1 => Some(BlockChecksumKind::Crc32c),
        #[cfg(feature = "checksum-rapidhash")]
        2 => Some(BlockChecksumKind::RapidHashV3_64),
        _ => None,
    }
}

pub(crate) fn create_options_with_key_len(key_len: usize) -> CreateOptions {
    CreateOptions::new(key_len, metadata(), test_block_checksum())
        .expect("test key length should be valid")
}

pub(crate) fn create_options_with_block_checksum(
    block_checksum: BlockChecksumKind,
) -> CreateOptions {
    CreateOptions::new(16, metadata(), block_checksum).expect("test key length should be valid")
}

pub(crate) fn create_options() -> CreateOptions {
    create_options_with_key_len(16)
}

pub(crate) fn open_options() -> StoreOpenOptions {
    StoreOpenOptions::read_write(metadata())
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
    Store::open(tempdir.path(), StoreOpenOptions::read_only(metadata()))
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

pub(crate) fn commit_entries(store: &Store, entries: &[(Vec<u8>, Vec<u8>)]) -> Result<CommitStats> {
    let mut batch = WriteBatch::new();
    for (key, value) in entries {
        batch.push(key, value);
    }
    store.commit_batch_with_options(batch, &commit_options())
}

pub(crate) fn commit_entries_with_options(
    store: &Store,
    entries: &[(Vec<u8>, Vec<u8>)],
    options: &CommitOptions,
) -> Result<CommitStats> {
    let mut batch = WriteBatch::new();
    for (key, value) in entries {
        batch.push(key, value);
    }
    store.commit_batch_with_options(batch, options)
}
