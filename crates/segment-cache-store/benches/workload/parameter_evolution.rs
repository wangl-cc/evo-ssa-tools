use std::hint::black_box;

use criterion::{BatchSize, Criterion, Throughput};

use crate::{
    backends::{
        create_segment_store, fill_segment_store, rebuild_segment_store_into,
        run_segment_axis_changes, sum_segment_fetches, touch_bytes,
    },
    data::{
        AxisChangeDataset, MiddleInsertDataset, build_axis_change_dataset,
        build_middle_insert_dataset,
    },
    profile::{PROFILES, ValueProfile},
};

const MIDDLE_INSERT_CHUNK: usize = 1024;

pub(crate) fn workload(c: &mut Criterion) {
    for &profile in PROFILES {
        let middle_insert_dataset = build_middle_insert_dataset(profile);
        bench_middle_insert_then_read(c, profile, &middle_insert_dataset);

        if matches!(profile, ValueProfile::Small) {
            let axis_change_dataset = build_axis_change_dataset(profile);
            bench_axis_change_rounds(c, profile, &axis_change_dataset);
        }
    }
}

fn bench_middle_insert_then_read(
    c: &mut Criterion,
    profile: ValueProfile,
    dataset: &MiddleInsertDataset,
) {
    let mut group = c.benchmark_group(format!("{}/middle_insert_then_read", profile.name()));
    group.throughput(Throughput::Elements(dataset.new_keys.len() as u64));

    group.bench_function("rebuild_new_store_then_read", |b| {
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

    group.bench_function("l0_chunked_insert_then_read", |b| {
        b.iter_batched(
            || {
                let tempdir = tempfile::tempdir().expect("tempdir should work");
                let store = create_segment_store(tempdir.path(), profile);
                fill_segment_store(&store, &dataset.old_entries);
                (tempdir, store)
            },
            |(_tempdir, store)| {
                let mut score = 0usize;
                for chunk in dataset.inserted_entries.chunks(MIDDLE_INSERT_CHUNK) {
                    let mut batch = store.begin_batch();
                    for (key, value) in chunk {
                        batch.push(key, value).expect("push should succeed");
                        score = score.wrapping_add(touch_bytes(value));
                    }
                    let stats = store.commit_batch(batch).expect("commit should succeed");
                    score = score
                        .wrapping_add(stats.records)
                        .wrapping_add(stats.merged_records)
                        .wrapping_add(stats.segments_published)
                        .wrapping_add(stats.segments_retired);
                }
                black_box(score.wrapping_add(sum_segment_fetches(&store, &dataset.new_keys)));
            },
            BatchSize::LargeInput,
        )
    });

    group.finish();
}

fn bench_axis_change_rounds(c: &mut Criterion, profile: ValueProfile, dataset: &AxisChangeDataset) {
    let report = dry_run_axis_change_rounds(profile, dataset);

    let mut group = c.benchmark_group(format!("{}/axis_change_rounds", profile.name()));
    group.throughput(Throughput::Elements(dataset.total_queries as u64));

    group.bench_function("lookup_commit", |b| {
        b.iter_batched(
            || tempfile::tempdir().expect("tempdir should work"),
            |tempdir| {
                black_box(run_segment_axis_changes(tempdir.path(), profile, dataset).score());
            },
            BatchSize::LargeInput,
        )
    });

    group.finish();

    eprintln!(
        "{}/axis_change_rounds dry-run: queries={} hits={} misses={} inserted={} merged_records={} rewrite_amp={:.2} retired={} published={}",
        profile.name(),
        report.queries,
        report.hits,
        report.misses,
        report.inserted,
        report.merged_records,
        report.rewrite_amplification(),
        report.segments_retired,
        report.segments_published
    );
}

fn dry_run_axis_change_rounds(
    profile: ValueProfile,
    dataset: &AxisChangeDataset,
) -> crate::backends::AxisChangeReport {
    let tempdir = tempfile::tempdir().expect("tempdir should work");
    run_segment_axis_changes(tempdir.path(), profile, dataset)
}
