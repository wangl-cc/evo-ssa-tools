use std::hint::black_box;

use criterion::{Criterion, Throughput};
use segment_cache_store::{CommitOptions, Store};

use crate::{
    backends::{
        commit_options_with_block_size, create_segment_store, fill_segment_store_with_options,
        sum_segment_fetches, sum_segment_iter, sum_segment_owned_fetches,
    },
    data::{Dataset, build_dataset},
    profile::{PROFILES, ValueProfile},
};

const LARGE_BLOCK_SIZE_VARIANTS: &[usize] = &[16 * 1024, 64 * 1024, 256 * 1024, 512 * 1024];
const SPARSE_BLOCK_SIZE_VARIANTS: &[usize] = &[16 * 1024, 32 * 1024, 64 * 1024, 256 * 1024];
const OVERLAY_RECORD_COUNT: usize = 16_384;
const OVERLAY_PATCH_STRIDE: usize = 8;
const OVERLAY_PATCH_COUNTS: &[usize] = &[1, 2, 4, 8];

pub(crate) fn workload(c: &mut Criterion) {
    for &profile in PROFILES {
        let dataset = build_dataset(16_384, profile);
        bench_ordered_fetch(c, profile, &dataset);
        bench_sparse_ordered_fetch(
            c,
            profile,
            "clustered_sparse_ordered_fetch",
            &dataset.entries,
            &dataset.clustered_sparse_ordered_keys,
        );
        bench_sparse_ordered_fetch(
            c,
            profile,
            "random_sparse_ordered_fetch",
            &dataset.entries,
            &dataset.random_sparse_ordered_keys,
        );

        if profile.uses_large_value_tuning() {
            bench_large_fetch_api(c, profile, &dataset);
            bench_large_block_sweep(c, profile, &dataset);
        }
    }

    let profile = ValueProfile::Small;
    let dataset = build_dataset(OVERLAY_RECORD_COUNT, profile);
    bench_overlay_ordered_fetch(c, profile, &dataset);
    bench_overlay_iter_all(c, profile, &dataset);
}

fn bench_ordered_fetch(c: &mut Criterion, profile: ValueProfile, dataset: &Dataset) {
    let mut group = c.benchmark_group(format!("{}/ordered_fetch", profile.name()));
    group.throughput(Throughput::Elements(dataset.ordered_keys.len() as u64));

    let tempdir = tempfile::tempdir().expect("tempdir should work");
    let store = create_filled_store(
        tempdir.path(),
        profile,
        &dataset.entries,
        &commit_options_with_block_size(16 * 1024),
    );
    group.bench_function("default_block", |b| {
        b.iter(|| black_box(sum_segment_fetches(&store, &dataset.ordered_keys)))
    });

    group.finish();
}

fn bench_sparse_ordered_fetch(
    c: &mut Criterion,
    profile: ValueProfile,
    name: &str,
    entries: &[(Vec<u8>, Vec<u8>)],
    keys: &[Vec<u8>],
) {
    let mut group = c.benchmark_group(format!("{}/{name}", profile.name()));
    group.throughput(Throughput::Elements(keys.len() as u64));

    for &block_size in SPARSE_BLOCK_SIZE_VARIANTS {
        let tempdir = tempfile::tempdir().expect("tempdir should work");
        let store = create_filled_store(
            tempdir.path(),
            profile,
            entries,
            &commit_options_with_block_size(block_size),
        );
        group.bench_function(format!("block_{}k", block_size / 1024), |b| {
            b.iter(|| black_box(sum_segment_fetches(&store, keys)))
        });
    }

    group.finish();
}

fn bench_large_block_sweep(c: &mut Criterion, profile: ValueProfile, dataset: &Dataset) {
    let mut group = c.benchmark_group(format!("{}/large_value_block_size", profile.name()));
    group.throughput(Throughput::Elements(dataset.ordered_keys.len() as u64));

    for &block_size in LARGE_BLOCK_SIZE_VARIANTS {
        let tempdir = tempfile::tempdir().expect("tempdir should work");
        let store = create_filled_store(
            tempdir.path(),
            profile,
            &dataset.entries,
            &commit_options_with_block_size(block_size),
        );
        group.bench_function(format!("block_{}k", block_size / 1024), |b| {
            b.iter(|| black_box(sum_segment_fetches(&store, &dataset.ordered_keys)))
        });
    }

    group.finish();
}

fn bench_large_fetch_api(c: &mut Criterion, profile: ValueProfile, dataset: &Dataset) {
    let mut group = c.benchmark_group(format!("{}/large_value_fetch_api", profile.name()));
    group.throughput(Throughput::Elements(dataset.ordered_keys.len() as u64));

    for &block_size in &[16 * 1024, 512 * 1024] {
        let tempdir = tempfile::tempdir().expect("tempdir should work");
        let store = create_filled_store(
            tempdir.path(),
            profile,
            &dataset.entries,
            &commit_options_with_block_size(block_size),
        );
        group.bench_function(format!("borrowed_block_{}k", block_size / 1024), |b| {
            b.iter(|| black_box(sum_segment_fetches(&store, &dataset.ordered_keys)))
        });
        group.bench_function(format!("owned_block_{}k", block_size / 1024), |b| {
            b.iter(|| black_box(sum_segment_owned_fetches(&store, &dataset.ordered_keys)))
        });
    }

    group.finish();
}

fn bench_overlay_ordered_fetch(c: &mut Criterion, profile: ValueProfile, dataset: &Dataset) {
    let overlay = OverlayReadDataset::from_dataset(dataset);
    let options = commit_options_with_block_size(16 * 1024);
    let main_only = OverlayStore::main_only(profile, &overlay, &options);
    let expected_checksum = sum_segment_fetches(&main_only.store, &overlay.ordered_keys);

    let overlays = OVERLAY_PATCH_COUNTS
        .iter()
        .copied()
        .map(|patch_count| {
            let store = OverlayStore::with_patch_count(profile, &overlay, &options, patch_count);
            assert_eq!(
                sum_segment_fetches(&store.store, &overlay.ordered_keys),
                expected_checksum,
                "overlay store should expose the same logical records as main-only"
            );
            (patch_count, store)
        })
        .collect::<Vec<_>>();

    let mut group = c.benchmark_group(format!("{}/overlay_ordered_fetch", profile.name()));
    group.throughput(Throughput::Elements(overlay.ordered_keys.len() as u64));

    group.bench_function("main_only", |b| {
        b.iter(|| black_box(sum_segment_fetches(&main_only.store, &overlay.ordered_keys)))
    });
    for (patch_count, overlay_store) in &overlays {
        group.bench_function(format!("patch_{patch_count}"), |b| {
            b.iter(|| {
                black_box(sum_segment_fetches(
                    &overlay_store.store,
                    &overlay.ordered_keys,
                ))
            })
        });
    }

    group.finish();
}

fn bench_overlay_iter_all(c: &mut Criterion, profile: ValueProfile, dataset: &Dataset) {
    let overlay = OverlayReadDataset::from_dataset(dataset);
    let options = commit_options_with_block_size(16 * 1024);
    let main_only = OverlayStore::main_only(profile, &overlay, &options);
    let expected_checksum = sum_segment_iter(&main_only.store);

    let overlays = OVERLAY_PATCH_COUNTS
        .iter()
        .copied()
        .map(|patch_count| {
            let store = OverlayStore::with_patch_count(profile, &overlay, &options, patch_count);
            assert_eq!(
                sum_segment_iter(&store.store),
                expected_checksum,
                "overlay store should scan the same logical records as main-only"
            );
            (patch_count, store)
        })
        .collect::<Vec<_>>();

    let mut group = c.benchmark_group(format!("{}/overlay_iter_all", profile.name()));
    group.throughput(Throughput::Elements(overlay.all_entries.len() as u64));

    group.bench_function("main_only", |b| {
        b.iter(|| black_box(sum_segment_iter(&main_only.store)))
    });
    for (patch_count, overlay_store) in &overlays {
        group.bench_function(format!("patch_{patch_count}"), |b| {
            b.iter(|| black_box(sum_segment_iter(&overlay_store.store)))
        });
    }

    group.finish();
}

fn create_filled_store(
    root: &std::path::Path,
    profile: ValueProfile,
    entries: &[(Vec<u8>, Vec<u8>)],
    options: &CommitOptions,
) -> Store {
    let store = create_segment_store(root, profile);
    fill_segment_store_with_options(&store, entries, options);
    store
}

struct OverlayReadDataset {
    ordered_keys: Vec<Vec<u8>>,
    all_entries: Vec<(Vec<u8>, Vec<u8>)>,
    main_entries: Vec<(Vec<u8>, Vec<u8>)>,
    patch_entries: Vec<(Vec<u8>, Vec<u8>)>,
}

impl OverlayReadDataset {
    fn from_dataset(dataset: &Dataset) -> Self {
        let mut main_entries = Vec::new();
        let mut patch_entries = Vec::new();
        for (index, entry) in dataset.entries.iter().cloned().enumerate() {
            if index % OVERLAY_PATCH_STRIDE == OVERLAY_PATCH_STRIDE / 2 {
                patch_entries.push(entry);
            } else {
                main_entries.push(entry);
            }
        }
        assert!(
            !patch_entries.is_empty(),
            "overlay benchmark needs patch entries"
        );
        Self {
            ordered_keys: dataset.ordered_keys.clone(),
            all_entries: dataset.entries.clone(),
            main_entries,
            patch_entries,
        }
    }
}

struct OverlayStore {
    _tempdir: tempfile::TempDir,
    store: Store,
}

impl OverlayStore {
    fn main_only(
        profile: ValueProfile,
        dataset: &OverlayReadDataset,
        options: &CommitOptions,
    ) -> Self {
        let tempdir = tempfile::tempdir().expect("tempdir should work");
        let store = create_segment_store(tempdir.path(), profile);
        fill_segment_store_with_options(&store, &dataset.all_entries, options);
        Self {
            _tempdir: tempdir,
            store,
        }
    }

    fn with_patch_count(
        profile: ValueProfile,
        dataset: &OverlayReadDataset,
        options: &CommitOptions,
        patch_count: usize,
    ) -> Self {
        assert!(patch_count > 0, "patch count should be non-zero");
        assert!(
            patch_count <= dataset.patch_entries.len(),
            "patch count should not exceed patch entries"
        );

        let tempdir = tempfile::tempdir().expect("tempdir should work");
        let store = create_segment_store(tempdir.path(), profile);
        fill_segment_store_with_options(&store, &dataset.main_entries, options);

        for chunk in chunk_by_count(&dataset.patch_entries, patch_count) {
            let mut batch = store.begin_batch();
            for (key, value) in chunk {
                batch.push(key, value).expect("push should succeed");
            }
            let stats = store
                .commit_batch_with_options(batch, options)
                .expect("patch commit should succeed");
            assert_eq!(stats.input_records, chunk.len());
            assert_eq!(stats.output_records, chunk.len());
            assert_eq!(stats.segments_retired, 0);
        }

        Self {
            _tempdir: tempdir,
            store,
        }
    }
}

fn chunk_by_count<T>(items: &[T], chunk_count: usize) -> impl Iterator<Item = &[T]> {
    let chunk_len = items.len().div_ceil(chunk_count);
    items.chunks(chunk_len)
}
