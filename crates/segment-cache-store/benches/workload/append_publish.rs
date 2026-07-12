use std::num::NonZeroUsize;

use criterion::{BatchSize, Criterion, Throughput};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};

use crate::{
    backends::{
        commit_options_with_block_size, create_segment_store, fill_segment_store,
        fill_segment_store_with_options,
    },
    data::{Dataset, build_dataset},
    profile::{PROFILES, ValueProfile},
};

pub(crate) fn workload(c: &mut Criterion) {
    for &profile in PROFILES {
        let dataset = build_dataset(16_384, profile);
        bench_append_publish(c, profile, &dataset);
    }
    bench_many_segment_publish(c);
}

fn bench_many_segment_publish(c: &mut Criterion) {
    const RECORD_COUNT: usize = 4_096;
    const RECORDS_PER_SEGMENT: usize = 128;

    let profile = ValueProfile::Small;
    let dataset = build_dataset(RECORD_COUNT, profile);
    let options = commit_options_with_block_size(16 * 1024)
        .with_flush_threshold_records(
            NonZeroUsize::new(RECORDS_PER_SEGMENT).expect("threshold is non-zero"),
        )
        .with_flush_threshold_bytes(NonZeroUsize::new(1024 * 1024).expect("threshold is non-zero"));
    let mut group = c.benchmark_group("small/append_publish_many_segments");
    group.throughput(Throughput::Elements(RECORD_COUNT as u64));
    group.bench_function("segments_32", |b| {
        b.iter_batched(
            || tempfile::tempdir().expect("tempdir should work"),
            |tempdir| {
                let store = create_segment_store(tempdir.path(), profile);
                fill_segment_store_with_options(&store, &dataset.entries, &options);
            },
            BatchSize::LargeInput,
        )
    });
    group.finish();
}

fn bench_append_publish(c: &mut Criterion, profile: ValueProfile, dataset: &Dataset) {
    let mut group = c.benchmark_group(format!("{}/append_publish", profile.name()));
    group.throughput(Throughput::Elements(dataset.entries.len() as u64));

    group.bench_function("sorted_batch", |b| {
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
    group.bench_function("unsorted_batch", |b| {
        b.iter_batched(
            || tempfile::tempdir().expect("tempdir should work"),
            |tempdir| {
                let store = create_segment_store(tempdir.path(), profile);
                let mut batch = segment_cache_store::WriteBatch::new();
                for (key, value) in &shuffled {
                    batch.push(key, value);
                }
                store.commit_batch(batch).expect("commit should succeed");
            },
            BatchSize::LargeInput,
        )
    });

    group.finish();
}
