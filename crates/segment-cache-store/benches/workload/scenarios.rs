use std::hint::black_box;

use criterion::{BatchSize, Criterion, Throughput};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use segment_cache_store::Store;

use crate::{
    backends::{
        Fjall3Backend, RedbBackend, commit_options_with_block_size, create_filled_segment_store,
        create_segment_store, fill_segment_store, profile_store_create_options,
        rebuild_segment_store_into, run_segment_axis_changes, sum_segment_fetches,
        sum_segment_iter,
    },
    data::{
        AxisChangeDataset, Dataset, MiddleInsertDataset, build_axis_change_dataset, build_dataset,
        build_middle_insert_dataset,
    },
    profile::{PROFILES, ValueProfile},
};

const LARGE_BLOCK_SIZE_VARIANTS: &[usize] = &[64 * 1024, 256 * 1024, 512 * 1024, 1024 * 1024];
const SPARSE_BLOCK_SIZE_VARIANTS: &[usize] = &[16 * 1024, 32 * 1024, 64 * 1024, 256 * 1024];

pub(crate) fn workload(c: &mut Criterion) {
    for &profile in PROFILES {
        let dataset = build_dataset(16_384, profile);
        bench_ordered_fetch(c, profile, &dataset);
        bench_sparse_ordered_fetch(c, profile, &dataset);
        bench_write_batch(c, profile, &dataset);
        bench_iter_all(c, profile, &dataset);
        let middle_insert_dataset = build_middle_insert_dataset(profile);
        bench_middle_insert_then_read(c, profile, &middle_insert_dataset);
        if matches!(profile, ValueProfile::Small) {
            let axis_change_dataset = build_axis_change_dataset(profile);
            bench_axis_change_rounds(c, profile, &axis_change_dataset);
        }
    }
}

fn bench_ordered_fetch(c: &mut Criterion, profile: ValueProfile, dataset: &Dataset) {
    let mut group = c.benchmark_group(format!("{}/ordered_fetch", profile.name()));
    group.throughput(Throughput::Elements(dataset.ordered_keys.len() as u64));

    let tempdir = tempfile::tempdir().expect("tempdir should work");
    let store = create_filled_segment_store(
        tempdir.path(),
        profile,
        &dataset.entries,
        &commit_options_with_block_size(16 * 1024),
        true,
    );
    group.bench_function("segment", |b| {
        b.iter(|| black_box(sum_segment_fetches(&store, &dataset.ordered_keys)))
    });
    let tempdir = tempfile::tempdir().expect("tempdir should work");
    let store = create_filled_segment_store(
        tempdir.path(),
        profile,
        &dataset.entries,
        &commit_options_with_block_size(16 * 1024),
        false,
    );
    group.bench_function("segment_no_crc", |b| {
        b.iter(|| black_box(sum_segment_fetches(&store, &dataset.ordered_keys)))
    });
    if profile.uses_large_value_tuning() {
        for &block_size in LARGE_BLOCK_SIZE_VARIANTS {
            let tempdir = tempfile::tempdir().expect("tempdir should work");
            let store = create_filled_segment_store(
                tempdir.path(),
                profile,
                &dataset.entries,
                &commit_options_with_block_size(block_size),
                true,
            );
            group.bench_function(format!("segment_block_{}k", block_size / 1024), |b| {
                b.iter(|| black_box(sum_segment_fetches(&store, &dataset.ordered_keys)))
            });

            let tempdir = tempfile::tempdir().expect("tempdir should work");
            let store = create_filled_segment_store(
                tempdir.path(),
                profile,
                &dataset.entries,
                &commit_options_with_block_size(block_size),
                false,
            );
            group.bench_function(
                format!("segment_block_{}k_no_crc", block_size / 1024),
                |b| b.iter(|| black_box(sum_segment_fetches(&store, &dataset.ordered_keys))),
            );
        }
    }
    if profile.fixed_value_len().is_some() {
        let tempdir = tempfile::tempdir().expect("tempdir should work");
        let store = create_filled_segment_store(
            tempdir.path(),
            profile,
            &dataset.entries,
            &commit_options_with_block_size(16 * 1024),
            true,
        );
        group.bench_function("segment_fixed_layout", |b| {
            b.iter(|| black_box(sum_segment_fetches(&store, &dataset.ordered_keys)))
        });
        let tempdir = tempfile::tempdir().expect("tempdir should work");
        let store = create_filled_segment_store(
            tempdir.path(),
            profile,
            &dataset.entries,
            &commit_options_with_block_size(16 * 1024),
            false,
        );
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

fn bench_sparse_ordered_fetch(c: &mut Criterion, profile: ValueProfile, dataset: &Dataset) {
    let mut group = c.benchmark_group(format!("{}/sparse_ordered_fetch", profile.name()));
    group.throughput(Throughput::Elements(
        dataset.sparse_ordered_keys.len() as u64
    ));

    for &block_size in SPARSE_BLOCK_SIZE_VARIANTS {
        let tempdir = tempfile::tempdir().expect("tempdir should work");
        let store = create_filled_segment_store(
            tempdir.path(),
            profile,
            &dataset.entries,
            &commit_options_with_block_size(block_size),
            true,
        );
        group.bench_function(format!("segment_block_{}k", block_size / 1024), |b| {
            b.iter(|| black_box(sum_segment_fetches(&store, &dataset.sparse_ordered_keys)))
        });
    }

    let tempdir = tempfile::tempdir().expect("tempdir should work");
    let fjall = Fjall3Backend::open(tempdir.path(), profile);
    fjall.fill(&dataset.entries);
    group.bench_function("fjall3", |b| {
        b.iter(|| black_box(fjall.sum_fetches(&dataset.sparse_ordered_keys)))
    });

    let tempdir = tempfile::tempdir().expect("tempdir should work");
    let redb = RedbBackend::open(tempdir.path());
    redb.fill(&dataset.entries);
    group.bench_function("redb", |b| {
        b.iter(|| black_box(redb.sum_fetches(&dataset.sparse_ordered_keys)))
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
                let store = create_segment_store(tempdir.path(), profile);
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
                let store = create_segment_store(tempdir.path(), profile);
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
                    let store = Store::create(
                        tempdir.path(),
                        profile_store_create_options(profile).with_fixed_value_len(value_len),
                    )
                    .expect("store should create");
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
    let store = create_filled_segment_store(
        tempdir.path(),
        profile,
        &dataset.entries,
        &commit_options_with_block_size(16 * 1024),
        true,
    );
    group.bench_function("segment", |b| {
        b.iter(|| black_box(sum_segment_iter(&store)))
    });
    let tempdir = tempfile::tempdir().expect("tempdir should work");
    let store = create_filled_segment_store(
        tempdir.path(),
        profile,
        &dataset.entries,
        &commit_options_with_block_size(16 * 1024),
        false,
    );
    group.bench_function("segment_no_crc", |b| {
        b.iter(|| black_box(sum_segment_iter(&store)))
    });
    if profile.uses_large_value_tuning() {
        for &block_size in LARGE_BLOCK_SIZE_VARIANTS {
            let tempdir = tempfile::tempdir().expect("tempdir should work");
            let store = create_filled_segment_store(
                tempdir.path(),
                profile,
                &dataset.entries,
                &commit_options_with_block_size(block_size),
                true,
            );
            group.bench_function(format!("segment_block_{}k", block_size / 1024), |b| {
                b.iter(|| black_box(sum_segment_iter(&store)))
            });

            let tempdir = tempfile::tempdir().expect("tempdir should work");
            let store = create_filled_segment_store(
                tempdir.path(),
                profile,
                &dataset.entries,
                &commit_options_with_block_size(block_size),
                false,
            );
            group.bench_function(
                format!("segment_block_{}k_no_crc", block_size / 1024),
                |b| b.iter(|| black_box(sum_segment_iter(&store))),
            );
        }
    }
    if profile.fixed_value_len().is_some() {
        let tempdir = tempfile::tempdir().expect("tempdir should work");
        let store = create_filled_segment_store(
            tempdir.path(),
            profile,
            &dataset.entries,
            &commit_options_with_block_size(16 * 1024),
            true,
        );
        group.bench_function("segment_fixed_layout", |b| {
            b.iter(|| black_box(sum_segment_iter(&store)))
        });
        let tempdir = tempfile::tempdir().expect("tempdir should work");
        let store = create_filled_segment_store(
            tempdir.path(),
            profile,
            &dataset.entries,
            &commit_options_with_block_size(16 * 1024),
            false,
        );
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
                let old_store = create_segment_store(old_tempdir.path(), profile);
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

fn bench_axis_change_rounds(c: &mut Criterion, profile: ValueProfile, dataset: &AxisChangeDataset) {
    validate_axis_change_rounds(profile, dataset);

    let mut group = c.benchmark_group(format!("{}/axis_change_rounds", profile.name()));
    group.throughput(Throughput::Elements(dataset.total_queries as u64));

    group.bench_function("segment_lookup_commit", |b| {
        b.iter_batched(
            || tempfile::tempdir().expect("tempdir should work"),
            |tempdir| {
                black_box(run_segment_axis_changes(tempdir.path(), profile, dataset).score());
            },
            BatchSize::LargeInput,
        )
    });

    group.bench_function("fjall3_lookup_insert", |b| {
        b.iter_batched(
            || tempfile::tempdir().expect("tempdir should work"),
            |tempdir| {
                let fjall = Fjall3Backend::open(tempdir.path(), profile);
                black_box(fjall.run_axis_changes(dataset).score());
            },
            BatchSize::LargeInput,
        )
    });

    group.bench_function("redb_lookup_insert", |b| {
        b.iter_batched(
            || tempfile::tempdir().expect("tempdir should work"),
            |tempdir| {
                let redb = RedbBackend::open(tempdir.path());
                black_box(redb.run_axis_changes(dataset).score());
            },
            BatchSize::LargeInput,
        )
    });

    group.finish();
}

fn validate_axis_change_rounds(profile: ValueProfile, dataset: &AxisChangeDataset) {
    let tempdir = tempfile::tempdir().expect("tempdir should work");
    let segment = run_segment_axis_changes(tempdir.path(), profile, dataset);

    let tempdir = tempfile::tempdir().expect("tempdir should work");
    let fjall = Fjall3Backend::open(tempdir.path(), profile);
    let fjall = fjall.run_axis_changes(dataset);

    let tempdir = tempfile::tempdir().expect("tempdir should work");
    let redb = RedbBackend::open(tempdir.path());
    let redb = redb.run_axis_changes(dataset);

    assert_eq!(segment.queries, fjall.queries);
    assert_eq!(segment.queries, redb.queries);
    assert_eq!(segment.hits, fjall.hits);
    assert_eq!(segment.hits, redb.hits);
    assert_eq!(segment.misses, fjall.misses);
    assert_eq!(segment.misses, redb.misses);
    assert_eq!(segment.inserted, fjall.inserted);
    assert_eq!(segment.inserted, redb.inserted);
    assert_eq!(segment.checksum, fjall.checksum);
    assert_eq!(segment.checksum, redb.checksum);

    eprintln!(
        "{}/axis_change_rounds dry-run: queries={} hits={} misses={} inserted={} merged_records={} rewrite_amp={:.2} retired={} published={}",
        profile.name(),
        segment.queries,
        segment.hits,
        segment.misses,
        segment.inserted,
        segment.merged_records,
        segment.rewrite_amplification(),
        segment.segments_retired,
        segment.segments_published
    );
}
