use std::hint::black_box;

use criterion::{BatchSize, Criterion, Throughput};

use crate::{
    backends::{
        Fjall3Backend, RedbBackend, commit_options_with_block_size, create_filled_segment_store,
        create_segment_store, fill_segment_store, run_segment_axis_changes, sum_segment_fetches,
        sum_segment_iter,
    },
    data::{AxisChangeDataset, Dataset, build_axis_change_dataset, build_dataset},
    profile::{PROFILES, ValueProfile},
};

pub(crate) fn workload(c: &mut Criterion) {
    for &profile in PROFILES {
        let dataset = build_dataset(16_384, profile);
        bench_ordered_fetch(c, profile, &dataset);
        bench_sparse_ordered_fetch(c, profile, &dataset);
        bench_iter_all(c, profile, &dataset);
        bench_append_commit(c, profile, &dataset);
    }

    let profile = ValueProfile::Small;
    let axis_change_dataset = build_axis_change_dataset(profile);
    bench_axis_change_rounds(c, profile, &axis_change_dataset);
}

fn bench_ordered_fetch(c: &mut Criterion, profile: ValueProfile, dataset: &Dataset) {
    let mut group = c.benchmark_group(format!("{}/comparison_ordered_fetch", profile.name()));
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
    let mut group = c.benchmark_group(format!(
        "{}/comparison_sparse_ordered_fetch",
        profile.name()
    ));
    group.throughput(Throughput::Elements(
        dataset.sparse_ordered_keys.len() as u64
    ));

    let tempdir = tempfile::tempdir().expect("tempdir should work");
    let store = create_filled_segment_store(
        tempdir.path(),
        profile,
        &dataset.entries,
        &commit_options_with_block_size(16 * 1024),
        true,
    );
    group.bench_function("segment", |b| {
        b.iter(|| black_box(sum_segment_fetches(&store, &dataset.sparse_ordered_keys)))
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
        b.iter(|| black_box(sum_segment_fetches(&store, &dataset.sparse_ordered_keys)))
    });

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

fn bench_iter_all(c: &mut Criterion, profile: ValueProfile, dataset: &Dataset) {
    let mut group = c.benchmark_group(format!("{}/comparison_iter_all", profile.name()));
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

fn bench_append_commit(c: &mut Criterion, profile: ValueProfile, dataset: &Dataset) {
    let mut group = c.benchmark_group(format!("{}/comparison_append_publish", profile.name()));
    group.throughput(Throughput::Elements(dataset.entries.len() as u64));

    group.bench_function("segment", |b| {
        b.iter_batched(
            || tempfile::tempdir().expect("tempdir should work"),
            |tempdir| {
                let store = create_segment_store(tempdir.path(), profile);
                fill_segment_store(&store, &dataset.entries);
            },
            BatchSize::LargeInput,
        )
    });

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

fn bench_axis_change_rounds(c: &mut Criterion, profile: ValueProfile, dataset: &AxisChangeDataset) {
    validate_axis_change_rounds(profile, dataset);

    let mut group = c.benchmark_group(format!("{}/comparison_axis_change_rounds", profile.name()));
    group.throughput(Throughput::Elements(dataset.total_queries as u64));

    group.bench_function("segment", |b| {
        b.iter_batched(
            || tempfile::tempdir().expect("tempdir should work"),
            |tempdir| {
                black_box(run_segment_axis_changes(tempdir.path(), profile, dataset).score());
            },
            BatchSize::LargeInput,
        )
    });

    group.bench_function("fjall3", |b| {
        b.iter_batched(
            || tempfile::tempdir().expect("tempdir should work"),
            |tempdir| {
                let fjall = Fjall3Backend::open(tempdir.path(), profile);
                black_box(fjall.run_axis_changes(dataset).score());
            },
            BatchSize::LargeInput,
        )
    });

    group.bench_function("redb", |b| {
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
        "{}/comparison_axis_change_rounds dry-run: queries={} hits={} misses={} inserted={} merged_records={} rewrite_amp={:.2} retired={} published={}",
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
