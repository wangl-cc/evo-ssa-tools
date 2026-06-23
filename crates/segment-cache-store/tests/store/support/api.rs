use segment_cache_store::{
    BlockChecksumKind, CommitOptions, CommitStats, CreateOptions, OpenOptions as StoreOpenOptions,
    Result, Store, StoreMetadata,
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

pub(crate) fn create_options_with_key_len(key_len: usize) -> CreateOptions {
    CreateOptions::new_with_block_checksum(key_len, metadata(), test_block_checksum())
}

pub(crate) fn create_options() -> CreateOptions {
    create_options_with_key_len(16)
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
    if !sorted {
        assert!(
            entries_have_key_inversion(entries),
            "test helper sorted flag must match unsorted input"
        );
    }
    for (key, value) in entries {
        batch.push(key, value)?;
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
    if !sorted {
        assert!(
            entries_have_key_inversion(entries),
            "test helper sorted flag must match unsorted input"
        );
    }
    for (key, value) in entries {
        batch.push(key, value)?;
    }
    store.commit_batch_with_options(batch, options)
}

fn entries_have_key_inversion(entries: &[(Vec<u8>, Vec<u8>)]) -> bool {
    entries.windows(2).any(|window| window[0].0 > window[1].0)
}
