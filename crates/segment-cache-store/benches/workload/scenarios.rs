use std::hint::black_box;

use criterion::{BatchSize, Criterion, Throughput};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use segment_cache_store::Store;

use crate::{
    backends::{
        Fjall3Backend, RedbBackend, fill_segment_store, fixed_store_options, profile_store_options,
        rebuild_segment_store_into, store_options, store_options_with_block_size,
        sum_segment_fetches, sum_segment_iter, unchecked_fixed_store_options,
        unchecked_store_options, unchecked_store_options_with_block_size,
    },
    data::{Dataset, MiddleInsertDataset, build_dataset, build_middle_insert_dataset},
    profile::{KEY_LEN, PROFILES, ValueProfile},
};

const LARGE_BLOCK_SIZE_VARIANTS: &[usize] = &[64 * 1024, 256 * 1024, 512 * 1024, 1024 * 1024];

pub(crate) fn workload(c: &mut Criterion) {
    for &profile in PROFILES {
        let dataset = build_dataset(16_384, profile);
        bench_ordered_fetch(c, profile, &dataset);
        bench_write_batch(c, profile, &dataset);
        bench_iter_all(c, profile, &dataset);
        let middle_insert_dataset = build_middle_insert_dataset(profile);
        bench_middle_insert_then_read(c, profile, &middle_insert_dataset);
    }
}

fn bench_ordered_fetch(c: &mut Criterion, profile: ValueProfile, dataset: &Dataset) {
    let mut group = c.benchmark_group(format!("{}/ordered_fetch", profile.name()));
    group.throughput(Throughput::Elements(dataset.ordered_keys.len() as u64));

    let tempdir = tempfile::tempdir().expect("tempdir should work");
    let store = Store::open(store_options(tempdir.path(), KEY_LEN)).expect("store should open");
    fill_segment_store(&store, &dataset.entries);
    group.bench_function("segment", |b| {
        b.iter(|| black_box(sum_segment_fetches(&store, &dataset.ordered_keys)))
    });
    let tempdir = tempfile::tempdir().expect("tempdir should work");
    let store =
        Store::open(unchecked_store_options(tempdir.path(), KEY_LEN)).expect("store should open");
    fill_segment_store(&store, &dataset.entries);
    group.bench_function("segment_no_crc", |b| {
        b.iter(|| black_box(sum_segment_fetches(&store, &dataset.ordered_keys)))
    });
    if profile.uses_large_value_tuning() {
        for &block_size in LARGE_BLOCK_SIZE_VARIANTS {
            let tempdir = tempfile::tempdir().expect("tempdir should work");
            let store = Store::open(store_options_with_block_size(
                tempdir.path(),
                KEY_LEN,
                block_size,
            ))
            .expect("store should open");
            fill_segment_store(&store, &dataset.entries);
            group.bench_function(format!("segment_block_{}k", block_size / 1024), |b| {
                b.iter(|| black_box(sum_segment_fetches(&store, &dataset.ordered_keys)))
            });

            let tempdir = tempfile::tempdir().expect("tempdir should work");
            let store = Store::open(unchecked_store_options_with_block_size(
                tempdir.path(),
                KEY_LEN,
                block_size,
            ))
            .expect("store should open");
            fill_segment_store(&store, &dataset.entries);
            group.bench_function(
                format!("segment_block_{}k_no_crc", block_size / 1024),
                |b| b.iter(|| black_box(sum_segment_fetches(&store, &dataset.ordered_keys))),
            );
        }
    }
    if let Some(value_len) = profile.fixed_value_len() {
        let tempdir = tempfile::tempdir().expect("tempdir should work");
        let store = Store::open(fixed_store_options(tempdir.path(), KEY_LEN, value_len))
            .expect("store should open");
        fill_segment_store(&store, &dataset.entries);
        group.bench_function("segment_fixed_layout", |b| {
            b.iter(|| black_box(sum_segment_fetches(&store, &dataset.ordered_keys)))
        });
        let tempdir = tempfile::tempdir().expect("tempdir should work");
        let store = Store::open(unchecked_fixed_store_options(
            tempdir.path(),
            KEY_LEN,
            value_len,
        ))
        .expect("store should open");
        fill_segment_store(&store, &dataset.entries);
        group.bench_function("segment_fixed_layout_no_crc", |b| {
            b.iter(|| black_box(sum_segment_fetches(&store, &dataset.ordered_keys)))
        });
    }

    let tempdir = tempfile::tempdir().expect("tempdir should work");
    let fjall = Fjall3Backend::open(tempdir.path(), profile);
    fjall.fill(&dataset.entries);
    group.bench_function("fjall3", |b| {
        b.iter(|| black_box(fjall.sum_fetches(&dataset.ordered_keys)))
    });

    let tempdir = tempfile::tempdir().expect("tempdir should work");
    let redb = RedbBackend::open(tempdir.path());
    redb.fill(&dataset.entries);
    group.bench_function("redb", |b| {
        b.iter(|| black_box(redb.sum_fetches(&dataset.ordered_keys)))
    });
    group.finish();
}

fn bench_write_batch(c: &mut Criterion, profile: ValueProfile, dataset: &Dataset) {
    let mut group = c.benchmark_group(format!("{}/append_commit", profile.name()));
    group.throughput(Throughput::Elements(dataset.entries.len() as u64));

    group.bench_function("segment_sorted", |b| {
        b.iter_batched(
            || tempfile::tempdir().expect("tempdir should work"),
            |tempdir| {
                let store =
                    Store::open(store_options(tempdir.path(), KEY_LEN)).expect("store should open");
                fill_segment_store(&store, &dataset.entries);
            },
            BatchSize::LargeInput,
        )
    });

    let mut shuffled = dataset.entries.clone();
    let mut rng = StdRng::seed_from_u64(7);
    shuffled.shuffle(&mut rng);
    group.bench_function("segment_unsorted", |b| {
        b.iter_batched(
            || tempfile::tempdir().expect("tempdir should work"),
            |tempdir| {
                let store =
                    Store::open(store_options(tempdir.path(), KEY_LEN)).expect("store should open");
                let mut batch = store.begin_batch();
                for (key, value) in &shuffled {
                    batch.push(key, value).expect("push should succeed");
                }
                store.commit_batch(batch).expect("commit should succeed");
            },
            BatchSize::LargeInput,
        )
    });
    if let Some(value_len) = profile.fixed_value_len() {
        group.bench_function("segment_fixed_layout_sorted", |b| {
            b.iter_batched(
                || tempfile::tempdir().expect("tempdir should work"),
                |tempdir| {
                    let store =
                        Store::open(fixed_store_options(tempdir.path(), KEY_LEN, value_len))
                            .expect("store should open");
                    fill_segment_store(&store, &dataset.entries);
                },
                BatchSize::LargeInput,
            )
        });
    }

    group.bench_function("fjall3", |b| {
        b.iter_batched(
            || tempfile::tempdir().expect("tempdir should work"),
            |tempdir| {
                let fjall = Fjall3Backend::open(tempdir.path(), profile);
                fjall.fill(&dataset.entries);
            },
            BatchSize::LargeInput,
        )
    });
    group.bench_function("redb", |b| {
        b.iter_batched(
            || tempfile::tempdir().expect("tempdir should work"),
            |tempdir| {
                let redb = RedbBackend::open(tempdir.path());
                redb.fill(&dataset.entries);
            },
            BatchSize::LargeInput,
        )
    });
    group.finish();
}

fn bench_iter_all(c: &mut Criterion, profile: ValueProfile, dataset: &Dataset) {
    let mut group = c.benchmark_group(format!("{}/iter_all", profile.name()));
    group.throughput(Throughput::Elements(dataset.entries.len() as u64));

    let tempdir = tempfile::tempdir().expect("tempdir should work");
    let store = Store::open(store_options(tempdir.path(), KEY_LEN)).expect("store should open");
    fill_segment_store(&store, &dataset.entries);
    group.bench_function("segment", |b| {
        b.iter(|| black_box(sum_segment_iter(&store)))
    });
    let tempdir = tempfile::tempdir().expect("tempdir should work");
    let store =
        Store::open(unchecked_store_options(tempdir.path(), KEY_LEN)).expect("store should open");
    fill_segment_store(&store, &dataset.entries);
    group.bench_function("segment_no_crc", |b| {
        b.iter(|| black_box(sum_segment_iter(&store)))
    });
    if profile.uses_large_value_tuning() {
        for &block_size in LARGE_BLOCK_SIZE_VARIANTS {
            let tempdir = tempfile::tempdir().expect("tempdir should work");
            let store = Store::open(store_options_with_block_size(
                tempdir.path(),
                KEY_LEN,
                block_size,
            ))
            .expect("store should open");
            fill_segment_store(&store, &dataset.entries);
            group.bench_function(format!("segment_block_{}k", block_size / 1024), |b| {
                b.iter(|| black_box(sum_segment_iter(&store)))
            });

            let tempdir = tempfile::tempdir().expect("tempdir should work");
            let store = Store::open(unchecked_store_options_with_block_size(
                tempdir.path(),
                KEY_LEN,
                block_size,
            ))
            .expect("store should open");
            fill_segment_store(&store, &dataset.entries);
            group.bench_function(
                format!("segment_block_{}k_no_crc", block_size / 1024),
                |b| b.iter(|| black_box(sum_segment_iter(&store))),
            );
        }
    }
    if let Some(value_len) = profile.fixed_value_len() {
        let tempdir = tempfile::tempdir().expect("tempdir should work");
        let store = Store::open(fixed_store_options(tempdir.path(), KEY_LEN, value_len))
            .expect("store should open");
        fill_segment_store(&store, &dataset.entries);
        group.bench_function("segment_fixed_layout", |b| {
            b.iter(|| black_box(sum_segment_iter(&store)))
        });
        let tempdir = tempfile::tempdir().expect("tempdir should work");
        let store = Store::open(unchecked_fixed_store_options(
            tempdir.path(),
            KEY_LEN,
            value_len,
        ))
        .expect("store should open");
        fill_segment_store(&store, &dataset.entries);
        group.bench_function("segment_fixed_layout_no_crc", |b| {
            b.iter(|| black_box(sum_segment_iter(&store)))
        });
    }

    let tempdir = tempfile::tempdir().expect("tempdir should work");
    let fjall = Fjall3Backend::open(tempdir.path(), profile);
    fjall.fill(&dataset.entries);
    group.bench_function("fjall3", |b| b.iter(|| black_box(fjall.sum_iter())));

    let tempdir = tempfile::tempdir().expect("tempdir should work");
    let redb = RedbBackend::open(tempdir.path());
    redb.fill(&dataset.entries);
    group.bench_function("redb", |b| b.iter(|| black_box(redb.sum_iter())));
    group.finish();
}

fn bench_middle_insert_then_read(
    c: &mut Criterion,
    profile: ValueProfile,
    dataset: &MiddleInsertDataset,
) {
    let mut group = c.benchmark_group(format!("{}/middle_insert_then_read", profile.name()));
    group.throughput(Throughput::Elements(dataset.new_keys.len() as u64));

    group.bench_function("segment_rebuild_new_store_then_read", |b| {
        b.iter_batched(
            || {
                let old_tempdir = tempfile::tempdir().expect("tempdir should work");
                let old_store = Store::open(profile_store_options(old_tempdir.path(), profile))
                    .expect("old store should open");
                fill_segment_store(&old_store, &dataset.old_entries);
                let new_tempdir = tempfile::tempdir().expect("tempdir should work");
                (old_tempdir, old_store, new_tempdir)
            },
            |(_old_tempdir, old_store, new_tempdir)| {
                let (new_store, rebuild_checksum) = rebuild_segment_store_into(
                    &old_store,
                    new_tempdir.path(),
                    profile,
                    &dataset.new_entries,
                );
                black_box(
                    rebuild_checksum
                        .wrapping_add(sum_segment_fetches(&new_store, &dataset.new_keys)),
                );
            },
            BatchSize::LargeInput,
        )
    });

    group.bench_function("fjall3_late_insert_then_read", |b| {
        b.iter_batched(
            || {
                let tempdir = tempfile::tempdir().expect("tempdir should work");
                let fjall = Fjall3Backend::open(tempdir.path(), profile);
                fjall.fill(&dataset.old_entries);
                (tempdir, fjall)
            },
            |(_tempdir, fjall)| {
                fjall.fill(&dataset.inserted_entries);
                black_box(fjall.sum_fetches(&dataset.new_keys));
            },
            BatchSize::LargeInput,
        )
    });

    group.bench_function("fjall3_late_insert_compact_then_read", |b| {
        b.iter_batched(
            || {
                let tempdir = tempfile::tempdir().expect("tempdir should work");
                let fjall = Fjall3Backend::open(tempdir.path(), profile);
                fjall.fill(&dataset.old_entries);
                (tempdir, fjall)
            },
            |(_tempdir, fjall)| {
                fjall.fill(&dataset.inserted_entries);
                fjall.major_compact();
                black_box(fjall.sum_fetches(&dataset.new_keys));
            },
            BatchSize::LargeInput,
        )
    });

    group.bench_function("redb_late_insert_then_read", |b| {
        b.iter_batched(
            || {
                let tempdir = tempfile::tempdir().expect("tempdir should work");
                let redb = RedbBackend::open(tempdir.path());
                redb.fill(&dataset.old_entries);
                (tempdir, redb)
            },
            |(_tempdir, redb)| {
                redb.fill(&dataset.inserted_entries);
                black_box(redb.sum_fetches(&dataset.new_keys));
            },
            BatchSize::LargeInput,
        )
    });

    group.finish();
}
