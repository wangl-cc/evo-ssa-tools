use std::{num::NonZeroU32, path::Path};

use fjall3::{
    KeyspaceCreateOptions, KvSeparationOptions,
    config::{BlockSizePolicy, HashRatioPolicy},
};
use redb::{Database, ReadableDatabase, ReadableTable, TableDefinition};
use segment_cache_store::{CommitOptions, CreateOptions, OpenOptions, Store, StoreMetadata};

use crate::profile::{KEY_LEN, ValueProfile};

const REDB_TABLE: TableDefinition<&[u8], &[u8]> = TableDefinition::new("bench");

pub(crate) fn store_metadata() -> StoreMetadata {
    StoreMetadata::from_text("segment-cache-store-bench")
}

pub(crate) fn store_create_options(key_len: usize) -> CreateOptions {
    CreateOptions::new(key_len, store_metadata())
}

pub(crate) fn commit_options_with_block_size(block_size: usize) -> CommitOptions {
    CommitOptions::default().with_target_block_size(block_size)
}

pub(crate) fn fixed_store_create_options(key_len: usize, value_len: NonZeroU32) -> CreateOptions {
    store_create_options(key_len).with_fixed_value_len(value_len)
}

pub(crate) fn open_options(verify_crc: bool) -> OpenOptions {
    OpenOptions::new(store_metadata()).with_block_checksum_verification(verify_crc)
}

pub(crate) fn create_segment_store(root: &Path, profile: ValueProfile, verify_crc: bool) -> Store {
    let create_options = profile_store_create_options(profile);
    let store = Store::create(root, create_options).expect("segment store should create");
    if verify_crc {
        store
    } else {
        drop(store);
        Store::open(root, open_options(false)).expect("segment store should reopen")
    }
}

pub(crate) fn profile_store_create_options(profile: ValueProfile) -> CreateOptions {
    match profile.fixed_value_len() {
        Some(value_len) => fixed_store_create_options(KEY_LEN, value_len),
        None => store_create_options(KEY_LEN),
    }
}

pub(crate) fn fill_segment_store(store: &Store, entries: &[(Vec<u8>, Vec<u8>)]) {
    fill_segment_store_with_options(store, entries, &commit_options_with_block_size(16 * 1024));
}

pub(crate) fn fill_segment_store_with_options(
    store: &Store,
    entries: &[(Vec<u8>, Vec<u8>)],
    options: &CommitOptions,
) {
    let mut batch = store.begin_batch().mark_sorted();
    for (key, value) in entries {
        batch.push(key, value).expect("push should succeed");
    }
    store
        .commit_batch_with_options(batch, options)
        .expect("commit should succeed");
}

pub(crate) fn rebuild_segment_store_into(
    old_store: &Store,
    new_root: &Path,
    profile: ValueProfile,
    new_entries: &[(Vec<u8>, Vec<u8>)],
) -> (Store, usize) {
    let new_store = create_segment_store(new_root, profile, true);
    let mut batch = new_store.begin_batch().mark_sorted();
    let mut old_records = old_store
        .iter_all()
        .expect("old store should scan")
        .peekable();
    let mut checksum = 0usize;

    for (key, computed_value) in new_entries {
        while let Some(Ok((old_key, _))) = old_records.peek()
            && old_key.as_slice() < key.as_slice()
        {
            let _ = old_records.next();
        }

        if let Some(Ok((old_key, _))) = old_records.peek()
            && old_key.as_slice() == key.as_slice()
        {
            let (old_key, value) = old_records
                .next()
                .expect("peeked old record should exist")
                .expect("peeked old record should be readable");
            checksum = checksum.wrapping_add(touch_bytes(&value));
            batch
                .push_owned(old_key, value)
                .expect("push should succeed");
        } else {
            batch
                .push(key, computed_value)
                .expect("push should succeed");
            checksum = checksum.wrapping_add(touch_bytes(computed_value));
        }
    }

    let stats = new_store
        .commit_batch(batch)
        .expect("rebuild commit should succeed");
    (new_store, checksum.wrapping_add(stats.records))
}

pub(crate) fn sum_segment_fetches(store: &Store, keys: &[Vec<u8>]) -> usize {
    let mut total = 0usize;
    store
        .visit_many_ordered_slice(keys, |_, value| {
            if let Some(value) = value {
                total = total.wrapping_add(touch_bytes(value));
            }
        })
        .expect("fetch should succeed");
    total
}

pub(crate) fn sum_segment_iter(store: &Store) -> usize {
    let mut total = 0usize;
    store
        .visit_all(|_, value| {
            total = total.wrapping_add(touch_bytes(value));
        })
        .expect("iter should succeed");
    total
}

pub(crate) fn touch_bytes(bytes: &[u8]) -> usize {
    bytes.iter().fold(bytes.len(), |acc, byte| {
        acc.wrapping_add(usize::from(*byte))
    })
}

pub(crate) struct Fjall3Backend {
    db: fjall3::Database,
    keyspace: fjall3::Keyspace,
}

impl Fjall3Backend {
    pub(crate) fn open(root: &Path, profile: ValueProfile) -> Self {
        let db = fjall3::Database::builder(root)
            .cache_size(256 * 1024 * 1024)
            .max_cached_files(Some(1024))
            .open()
            .expect("fjall3 database should open");
        let keyspace = db
            .keyspace("bench", || fjall3_keyspace_options(profile))
            .expect("fjall3 keyspace should open");
        Self { db, keyspace }
    }

    pub(crate) fn fill(&self, entries: &[(Vec<u8>, Vec<u8>)]) {
        for (key, value) in entries {
            self.keyspace
                .insert(key, value)
                .expect("insert should work");
        }
        self.db
            .persist(fjall3::PersistMode::SyncAll)
            .expect("persist should succeed");
    }

    pub(crate) fn sum_fetches(&self, keys: &[Vec<u8>]) -> usize {
        keys.iter()
            .filter_map(|key| self.keyspace.get(key).expect("get should succeed"))
            .map(|value| touch_bytes(value.as_ref()))
            .sum()
    }

    pub(crate) fn sum_iter(&self) -> usize {
        self.keyspace
            .range::<&[u8], _>(..)
            .map(|entry| {
                let value = entry
                    .into_inner()
                    .expect("range entry should be readable")
                    .1;
                touch_bytes(value.as_ref())
            })
            .sum()
    }

    pub(crate) fn major_compact(&self) {
        self.keyspace
            .major_compact()
            .expect("fjall3 major compaction should succeed");
        self.db
            .persist(fjall3::PersistMode::SyncAll)
            .expect("fjall3 persist after compaction should succeed");
    }
}

pub(crate) struct RedbBackend {
    db: Database,
}

impl RedbBackend {
    pub(crate) fn open(root: &Path) -> Self {
        let db = Database::create(root.join("bench.redb")).expect("redb database should open");
        Self { db }
    }

    pub(crate) fn fill(&self, entries: &[(Vec<u8>, Vec<u8>)]) {
        let write_txn = self.db.begin_write().expect("redb write txn should open");
        {
            let mut table = write_txn
                .open_table(REDB_TABLE)
                .expect("redb table should open");
            for (key, value) in entries {
                table
                    .insert(key.as_slice(), value.as_slice())
                    .expect("redb insert should succeed");
            }
        }
        write_txn.commit().expect("redb commit should succeed");
    }

    pub(crate) fn sum_fetches(&self, keys: &[Vec<u8>]) -> usize {
        let read_txn = self.db.begin_read().expect("redb read txn should open");
        let table = read_txn
            .open_table(REDB_TABLE)
            .expect("redb table should open");
        keys.iter()
            .filter_map(|key| table.get(key.as_slice()).expect("redb get should succeed"))
            .map(|value| touch_bytes(value.value()))
            .sum()
    }

    pub(crate) fn sum_iter(&self) -> usize {
        let read_txn = self.db.begin_read().expect("redb read txn should open");
        let table = read_txn
            .open_table(REDB_TABLE)
            .expect("redb table should open");
        table
            .iter()
            .expect("redb iter should open")
            .map(|entry| {
                let (_key, value) = entry.expect("redb iter entry should be readable");
                touch_bytes(value.value())
            })
            .sum()
    }
}

fn fjall3_small_value_options() -> KeyspaceCreateOptions {
    KeyspaceCreateOptions::default()
        .expect_point_read_hits(true)
        .data_block_hash_ratio_policy(HashRatioPolicy::all(0.35))
        .data_block_size_policy(BlockSizePolicy::all(4 * 1024))
        .max_memtable_size(128 * 1024 * 1024)
}

fn fjall3_large_value_options() -> KeyspaceCreateOptions {
    KeyspaceCreateOptions::default()
        .expect_point_read_hits(true)
        .data_block_hash_ratio_policy(HashRatioPolicy::all(0.1))
        .data_block_size_policy(BlockSizePolicy::all(16 * 1024))
        .with_kv_separation(Some(KvSeparationOptions::default()))
        .max_memtable_size(64 * 1024 * 1024)
}

fn fjall3_keyspace_options(profile: ValueProfile) -> KeyspaceCreateOptions {
    if profile.uses_large_value_tuning() {
        fjall3_large_value_options()
    } else {
        fjall3_small_value_options()
    }
}
