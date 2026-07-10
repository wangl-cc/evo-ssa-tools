#![allow(
    dead_code,
    reason = "Criterion targets reuse benchmark fixtures selectively"
)]

use std::{num::NonZeroU32, path::Path};

use fjall3::{
    KeyspaceCreateOptions, KvSeparationOptions,
    config::{BlockSizePolicy, HashRatioPolicy},
};
use redb::{Database, ReadableDatabase, ReadableTable, TableDefinition};
use segment_cache_store::{CommitOptions, CreateOptions, OpenOptions, Store, StoreMetadata};

use crate::{
    data::AxisChangeDataset,
    profile::{KEY_LEN, ValueProfile},
};

const REDB_TABLE: TableDefinition<&[u8], &[u8]> = TableDefinition::new("bench");

pub(crate) fn store_metadata() -> StoreMetadata {
    StoreMetadata::from_text("segment-cache-store-bench")
}

pub(crate) fn store_create_options(key_len: usize) -> CreateOptions {
    #[cfg(feature = "checksum-rapidhash")]
    {
        CreateOptions::new(key_len, store_metadata())
    }
    #[cfg(not(feature = "checksum-rapidhash"))]
    {
        CreateOptions::new_with_block_checksum(
            key_len,
            store_metadata(),
            segment_cache_store::BlockChecksumKind::None,
        )
    }
}

pub(crate) fn commit_options_with_block_size(block_size: usize) -> CommitOptions {
    CommitOptions::default().with_target_block_size(block_size)
}

pub(crate) fn fixed_store_create_options(key_len: usize, value_len: NonZeroU32) -> CreateOptions {
    store_create_options(key_len).with_fixed_value_len(value_len)
}

pub(crate) fn open_options(verify_crc: bool) -> OpenOptions {
    OpenOptions::new(store_metadata())
        .with_block_checksum_verification(verify_crc)
        .with_read_only(!verify_crc)
}

pub(crate) fn create_segment_store(root: &Path, profile: ValueProfile) -> Store {
    let create_options = profile_store_create_options(profile);
    Store::create(root, create_options).expect("segment store should create")
}

pub(crate) fn open_segment_store(root: &Path, verify_crc: bool) -> Store {
    Store::open(root, open_options(verify_crc)).expect("segment store should reopen")
}

pub(crate) fn create_filled_segment_store(
    root: &Path,
    profile: ValueProfile,
    entries: &[(Vec<u8>, Vec<u8>)],
    options: &CommitOptions,
    verify_crc: bool,
) -> Store {
    let store = create_segment_store(root, profile);
    fill_segment_store_with_options(&store, entries, options);
    if verify_crc {
        store
    } else {
        drop(store);
        open_segment_store(root, false)
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
    let mut batch = store.begin_batch();
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
    let new_store = create_segment_store(new_root, profile);
    let mut batch = new_store.begin_batch();
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
    (new_store, checksum.wrapping_add(stats.input_records))
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

pub(crate) fn sum_segment_owned_fetches(store: &Store, keys: &[Vec<u8>]) -> usize {
    store
        .fetch_many_ordered(keys.iter().map(Vec::as_slice))
        .expect("fetch should succeed")
        .into_iter()
        .flatten()
        .map(|value| touch_bytes(&value))
        .sum()
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct AxisChangeReport {
    pub(crate) queries: usize,
    pub(crate) hits: usize,
    pub(crate) misses: usize,
    pub(crate) inserted: usize,
    pub(crate) output_records: usize,
    pub(crate) segments_published: usize,
    pub(crate) segments_retired: usize,
    pub(crate) checksum: usize,
}

impl AxisChangeReport {
    pub(crate) fn score(self) -> usize {
        self.checksum
            .wrapping_add(self.queries)
            .wrapping_add(self.hits)
            .wrapping_add(self.misses)
            .wrapping_add(self.inserted)
            .wrapping_add(self.output_records)
            .wrapping_add(self.segments_published)
            .wrapping_add(self.segments_retired)
    }

    pub(crate) fn rewrite_amplification(self) -> f64 {
        if self.inserted == 0 {
            0.0
        } else {
            self.output_records as f64 / self.inserted as f64
        }
    }
}

pub(crate) fn run_segment_axis_changes(
    root: &Path,
    profile: ValueProfile,
    dataset: &AxisChangeDataset,
) -> AxisChangeReport {
    let store = create_segment_store(root, profile);
    let mut report = AxisChangeReport::default();

    for round in &dataset.rounds {
        report.queries += round.entries.len();
        let mut missing_indices = Vec::new();
        store
            .visit_many_ordered_slice(&round.keys, |index, value| {
                if let Some(value) = value {
                    report.hits += 1;
                    report.checksum = report.checksum.wrapping_add(touch_bytes(value));
                } else {
                    report.misses += 1;
                    missing_indices.push(index);
                }
            })
            .expect("axis-change lookup should succeed");

        if missing_indices.is_empty() {
            continue;
        }
        let mut batch = store.begin_batch();
        for index in missing_indices {
            let (key, value) = &round.entries[index];
            report.checksum = report.checksum.wrapping_add(touch_bytes(value));
            batch.push(key, value).expect("push should succeed");
        }
        let stats = store.commit_batch(batch).expect("commit should succeed");
        report.inserted += stats.input_records;
        report.output_records += stats.output_records;
        report.segments_published += stats.segments_published;
        report.segments_retired += stats.segments_retired;
    }

    report
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

    pub(crate) fn run_axis_changes(&self, dataset: &AxisChangeDataset) -> AxisChangeReport {
        let mut report = AxisChangeReport::default();
        for round in &dataset.rounds {
            report.queries += round.entries.len();
            let mut wrote = false;
            for (key, value) in &round.entries {
                if let Some(existing) = self.keyspace.get(key).expect("get should succeed") {
                    report.hits += 1;
                    report.checksum = report.checksum.wrapping_add(touch_bytes(existing.as_ref()));
                } else {
                    report.misses += 1;
                    report.inserted += 1;
                    report.checksum = report.checksum.wrapping_add(touch_bytes(value));
                    self.keyspace
                        .insert(key, value)
                        .expect("insert should work");
                    wrote = true;
                }
            }
            if wrote {
                self.db
                    .persist(fjall3::PersistMode::SyncAll)
                    .expect("persist should succeed");
            }
        }
        report
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

    pub(crate) fn run_axis_changes(&self, dataset: &AxisChangeDataset) -> AxisChangeReport {
        self.create_table();
        let mut report = AxisChangeReport::default();
        for round in &dataset.rounds {
            report.queries += round.entries.len();
            let mut missing_indices = Vec::new();
            {
                let read_txn = self.db.begin_read().expect("redb read txn should open");
                let table = read_txn
                    .open_table(REDB_TABLE)
                    .expect("redb table should open");
                for (index, (key, _)) in round.entries.iter().enumerate() {
                    if let Some(value) = table.get(key.as_slice()).expect("redb get should succeed")
                    {
                        report.hits += 1;
                        report.checksum = report.checksum.wrapping_add(touch_bytes(value.value()));
                    } else {
                        report.misses += 1;
                        missing_indices.push(index);
                    }
                }
            }
            if missing_indices.is_empty() {
                continue;
            }
            let write_txn = self.db.begin_write().expect("redb write txn should open");
            {
                let mut table = write_txn
                    .open_table(REDB_TABLE)
                    .expect("redb table should open");
                for index in missing_indices {
                    let (key, value) = &round.entries[index];
                    report.inserted += 1;
                    report.checksum = report.checksum.wrapping_add(touch_bytes(value));
                    table
                        .insert(key.as_slice(), value.as_slice())
                        .expect("redb insert should succeed");
                }
            }
            write_txn.commit().expect("redb commit should succeed");
        }
        report
    }

    fn create_table(&self) {
        let write_txn = self.db.begin_write().expect("redb write txn should open");
        {
            write_txn
                .open_table(REDB_TABLE)
                .expect("redb table should open");
        }
        write_txn.commit().expect("redb commit should succeed");
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
