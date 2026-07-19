use std::{hint::black_box, path::Path};

use criterion::{BatchSize, Criterion, Throughput};
use segment_cache_store::{
    BlockChecksumKind, CommitOptions, CreateOptions, OpenOptions, Store, StoreMetadata, WriteBatch,
};

const OPEN_RECORD_COUNT: usize = 2_048;
const OPEN_VALUE_LEN: usize = 16 * 1024;
const ROUTING_RECORD_COUNT: usize = 16_384;
const ROUTING_VALUE_LEN: usize = 64;
const ROUTING_STRIDE: usize = 16;

pub(crate) fn workload(c: &mut Criterion) {
    bench_trusted_open(c);
    bench_key_width_routing(c);
}

fn bench_trusted_open(c: &mut Criterion) {
    let entries = build_entries(128, OPEN_RECORD_COUNT, OPEN_VALUE_LEN);
    let tempdir = tempfile::tempdir().expect("tempdir should work");
    let store = create_filled_store(tempdir.path(), 128, &entries, 256 * 1024);
    let stats = store.storage_stats().expect("storage stats should load");
    eprintln!(
        "physical_format_size/open_segment segment_bytes={}",
        stats.segment_bytes
    );
    drop(store);

    let mut group = c.benchmark_group("physical_format/trusted_open");
    group.throughput(Throughput::Bytes(stats.segment_bytes));
    group.bench_function("segment_32m", |b| {
        b.iter_batched(
            || (),
            |()| {
                black_box(
                    Store::open(tempdir.path(), OpenOptions::read_only(store_metadata()))
                        .expect("store should reopen"),
                )
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

fn bench_key_width_routing(c: &mut Criterion) {
    let mut group = c.benchmark_group("physical_format/sparse_routing");
    group.throughput(Throughput::Elements(
        (ROUTING_RECORD_COUNT / ROUTING_STRIDE) as u64,
    ));

    for key_len in [16, 512] {
        let entries = build_entries(key_len, ROUTING_RECORD_COUNT, ROUTING_VALUE_LEN);
        let keys = entries
            .iter()
            .step_by(ROUTING_STRIDE)
            .map(|(key, _)| key.clone())
            .collect::<Vec<_>>();
        let tempdir = tempfile::tempdir().expect("tempdir should work");
        let store = create_filled_store(tempdir.path(), key_len, &entries, 16 * 1024);
        let stats = store.storage_stats().expect("storage stats should load");
        eprintln!(
            "physical_format_size/key_{key_len} segment_bytes={}",
            stats.segment_bytes
        );

        group.bench_function(format!("key_{key_len}"), |b| {
            b.iter(|| {
                let mut checksum = 0usize;
                store
                    .visit_many_ordered(&keys, |_, value| {
                        if let Some(value) = value {
                            checksum = checksum.wrapping_add(value.len());
                        }
                    })
                    .expect("ordered lookup should succeed");
                black_box(checksum)
            })
        });
    }
    group.finish();
}

fn build_entries(key_len: usize, record_count: usize, value_len: usize) -> Vec<(Vec<u8>, Vec<u8>)> {
    (0..record_count)
        .map(|index| {
            let mut key = vec![b'x'; key_len];
            key[key_len - size_of::<u64>()..].copy_from_slice(&(index as u64).to_be_bytes());
            let mut value = vec![index as u8; value_len];
            value[..size_of::<u64>()].copy_from_slice(&(index as u64).to_le_bytes());
            (key, value)
        })
        .collect()
}

fn create_filled_store(
    root: &Path,
    key_len: usize,
    entries: &[(Vec<u8>, Vec<u8>)],
    block_size: u32,
) -> Store {
    let store = Store::create(
        root,
        CreateOptions::new(key_len, store_metadata(), benchmark_checksum())
            .expect("benchmark key length should be valid"),
    )
    .expect("store should create");
    let mut batch = WriteBatch::new();
    for (key, value) in entries {
        batch.push(key, value);
    }
    store
        .commit_batch_with_options(
            batch,
            &CommitOptions::default().with_target_block_size(block_size),
        )
        .expect("benchmark entries should commit");
    store
}

fn store_metadata() -> StoreMetadata {
    StoreMetadata::from_text("segment-cache-store-physical-format-bench")
}

fn benchmark_checksum() -> BlockChecksumKind {
    #[cfg(feature = "checksum-rapidhash")]
    {
        BlockChecksumKind::RapidHashV3_64
    }
    #[cfg(not(feature = "checksum-rapidhash"))]
    {
        BlockChecksumKind::None
    }
}
