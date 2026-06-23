use criterion::{BatchSize, Criterion, Throughput};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};

use crate::{
    backends::{create_segment_store, fill_segment_store},
    data::{Dataset, build_dataset},
    profile::{PROFILES, ValueProfile},
};

pub(crate) fn workload(c: &mut Criterion) {
    for &profile in PROFILES {
        let dataset = build_dataset(16_384, profile);
        bench_append_publish(c, profile, &dataset);
    }
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
                let mut batch = store.begin_batch();
                for (key, value) in &shuffled {
                    batch.push(key, value).expect("push should succeed");
                }
                store.commit_batch(batch).expect("commit should succeed");
            },
            BatchSize::LargeInput,
        )
    });

    group.finish();
}
