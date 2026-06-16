use std::{fs, num::NonZeroU32, path::Path};

use fjall3::{
    KeyspaceCreateOptions, KvSeparationOptions,
    config::{BlockSizePolicy, HashRatioPolicy},
};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use redb::{Database, TableDefinition};
use segment_cache_store::{CommitOptions, CreateOptions, Store, StoreMetadata};

const KEY_LEN: usize = 128;
const REDB_TABLE: TableDefinition<&[u8], &[u8]> = TableDefinition::new("bench");

#[derive(Clone, Copy)]
enum ValueProfile {
    Small,
    SmallFixed,
    Medium,
    Large,
}

impl ValueProfile {
    fn name(self) -> &'static str {
        match self {
            Self::Small => "small",
            Self::SmallFixed => "small_fixed",
            Self::Medium => "medium",
            Self::Large => "large",
        }
    }

    fn base_len(self) -> usize {
        match self {
            Self::Small => 64,
            Self::SmallFixed => 64,
            Self::Medium => 1_024,
            Self::Large => 16 * 1_024,
        }
    }

    fn jitter(self) -> usize {
        match self {
            Self::Small => 8,
            Self::SmallFixed => 0,
            Self::Medium => 128,
            Self::Large => 2 * 1_024,
        }
    }

    fn uses_large_value_tuning(self) -> bool {
        matches!(self, Self::Large)
    }

    fn fixed_value_len(self) -> Option<NonZeroU32> {
        match self {
            Self::SmallFixed => Some(
                NonZeroU32::new(
                    u32::try_from(self.base_len()).expect("profile value len should fit"),
                )
                .expect("profile value len is non-zero"),
            ),
            Self::Small | Self::Medium | Self::Large => None,
        }
    }
}

fn make_key(axis_0: u64, axis_1: u64, axis_2: u64, axis_3: u64, axis_4: u64, rep: u64) -> Vec<u8> {
    let fields = [
        0x5353475f43414348_u64,
        0x0000_0000_0000_0001,
        0x0000_0000_0000_0010,
        0x0000_0000_0000_0020,
        0x0000_0000_0000_0030,
        0x0000_0000_0000_0040,
        axis_0,
        axis_1,
        axis_2,
        axis_3,
        axis_4,
        0,
        0,
        0,
        0,
        rep,
    ];
    let mut key = Vec::with_capacity(KEY_LEN);
    for field in fields {
        key.extend_from_slice(&field.to_be_bytes());
    }
    debug_assert_eq!(key.len(), KEY_LEN);
    key
}

fn make_value(rng: &mut StdRng, profile: ValueProfile) -> Vec<u8> {
    let base = profile.base_len();
    let jitter = profile.jitter();
    let spread = jitter.saturating_mul(2) + 1;
    let offset = usize::from(rng.random::<u16>()) % spread;
    let len = base + offset - jitter;
    (0..len).map(|_| rng.random()).collect()
}

fn build_dataset(n: usize, profile: ValueProfile) -> Vec<(Vec<u8>, Vec<u8>)> {
    let mut rng = StdRng::seed_from_u64(
        42 + u64::try_from(profile.base_len()).expect("profile base len should fit"),
    );
    let mut entries = Vec::with_capacity(n);
    for index in 0..n {
        let axis_0 = (index / 4096) as u64;
        let axis_1 = ((index / 1024) % 4) as u64;
        let axis_2 = ((index / 256) % 4) as u64;
        let axis_3 = ((index / 16) % 16) as u64;
        let axis_4 = 0;
        let rep = (index % 16) as u64;
        entries.push((
            make_key(axis_0, axis_1, axis_2, axis_3, axis_4, rep),
            make_value(&mut rng, profile),
        ));
    }
    entries.sort_by(|left, right| left.0.cmp(&right.0));
    entries
}

fn logical_bytes(entries: &[(Vec<u8>, Vec<u8>)]) -> usize {
    entries
        .iter()
        .map(|(key, value)| key.len() + value.len())
        .sum()
}

fn dir_size(path: &Path) -> u64 {
    fn walk(path: &Path) -> u64 {
        let metadata = fs::symlink_metadata(path).expect("metadata should exist");
        if metadata.is_file() {
            return metadata.len();
        }
        if metadata.is_dir() {
            return fs::read_dir(path)
                .expect("dir should be readable")
                .map(|entry| walk(&entry.expect("dir entry should exist").path()))
                .sum();
        }
        0
    }

    walk(path)
}

fn write_segment_store(root: &Path, entries: &[(Vec<u8>, Vec<u8>)]) -> u64 {
    let store =
        Store::create(root, segment_create_options(None)).expect("segment store should create");
    write_entries_to_segment_store(&store, entries);
    dir_size(root)
}

fn write_fixed_segment_store(
    root: &Path,
    entries: &[(Vec<u8>, Vec<u8>)],
    value_len: NonZeroU32,
) -> u64 {
    let store = Store::create(root, segment_create_options(Some(value_len)))
        .expect("segment store should create");
    write_entries_to_segment_store(&store, entries);
    dir_size(root)
}

fn segment_create_options(fixed_value_len: Option<NonZeroU32>) -> CreateOptions {
    let options = default_segment_create_options();
    if let Some(value_len) = fixed_value_len {
        options.with_fixed_value_len(value_len)
    } else {
        options
    }
}

fn default_segment_create_options() -> CreateOptions {
    #[cfg(feature = "checksum-rapidhash")]
    {
        CreateOptions::new(KEY_LEN, StoreMetadata::from_text("space-usage"))
    }
    #[cfg(not(feature = "checksum-rapidhash"))]
    {
        CreateOptions::new_with_block_checksum(
            KEY_LEN,
            StoreMetadata::from_text("space-usage"),
            segment_cache_store::BlockChecksumKind::None,
        )
    }
}

fn write_entries_to_segment_store(store: &Store, entries: &[(Vec<u8>, Vec<u8>)]) {
    let mut batch = store.begin_batch().mark_sorted();
    for (key, value) in entries {
        batch.push(key, value).expect("push should succeed");
    }
    store
        .commit_batch_with_options(
            batch,
            &CommitOptions::default().with_target_block_size(16 * 1024),
        )
        .expect("commit should succeed");
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

fn write_fjall3(root: &Path, entries: &[(Vec<u8>, Vec<u8>)], profile: ValueProfile) -> u64 {
    let db = fjall3::Database::builder(root)
        .cache_size(256 * 1024 * 1024)
        .max_cached_files(Some(1024))
        .open()
        .expect("fjall3 database should open");
    let keyspace = db
        .keyspace("bench", || fjall3_keyspace_options(profile))
        .expect("fjall3 keyspace should open");
    for (key, value) in entries {
        keyspace.insert(key, value).expect("insert should succeed");
    }
    db.persist(fjall3::PersistMode::SyncAll)
        .expect("persist should succeed");
    drop(keyspace);
    drop(db);
    dir_size(root)
}

fn write_redb(root: &Path, entries: &[(Vec<u8>, Vec<u8>)]) -> u64 {
    fs::create_dir_all(root).expect("redb root should be creatable");
    let db = Database::create(root.join("bench.redb")).expect("redb database should open");
    let write_txn = db.begin_write().expect("redb write txn should open");
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
    drop(db);
    dir_size(root)
}

fn main() {
    let tempdir = tempfile::tempdir().expect("tempdir should be creatable");
    let profiles = [
        ValueProfile::Small,
        ValueProfile::SmallFixed,
        ValueProfile::Medium,
        ValueProfile::Large,
    ];

    for profile in profiles {
        let entries = build_dataset(16_384, profile);
        let logical = logical_bytes(&entries) as f64;
        let segment_size = write_segment_store(
            &tempdir.path().join(profile.name()).join("segment"),
            &entries,
        );
        let fixed_segment_size = profile.fixed_value_len().map(|value_len| {
            write_fixed_segment_store(
                &tempdir.path().join(profile.name()).join("segment-fixed"),
                &entries,
                value_len,
            )
        });
        let fjall3_size = write_fjall3(
            &tempdir.path().join(profile.name()).join("fjall3"),
            &entries,
            profile,
        );
        let redb_size = write_redb(&tempdir.path().join(profile.name()).join("redb"), &entries);

        println!(
            "{} logical={} segment={} ({:.3}x) fixed_segment={} fjall3={} ({:.3}x) redb={} ({:.3}x)",
            profile.name(),
            logical as u64,
            segment_size,
            segment_size as f64 / logical,
            fixed_segment_size
                .map(|size| format!("{size} ({:.3}x)", size as f64 / logical))
                .unwrap_or_else(|| "n/a".to_string()),
            fjall3_size,
            fjall3_size as f64 / logical,
            redb_size,
            redb_size as f64 / logical
        );
    }
}
