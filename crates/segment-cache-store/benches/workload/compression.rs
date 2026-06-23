use std::{hint::black_box, path::Path};

use criterion::{BatchSize, Criterion, Throughput};
use segment_cache_store::{CommitOptions, Store, ValuePayloadCompressionKind};

use crate::{
    backends::{
        commit_options_with_block_size, fill_segment_store_with_options,
        profile_store_create_options, sum_segment_fetches, sum_segment_iter,
    },
    data::{Dataset, ValueEntropy, build_compression_dataset},
    profile::ValueProfile,
};

const RECORD_COUNT: usize = 512;
const VALUE_LEN: usize = 128 * 1024;
const VALUE_JITTER: usize = 8 * 1024;
const WORKLOAD_NAME: &str = "value_128k";
const DEFAULT_BLOCK_SIZE: usize = 16 * 1024;

#[derive(Clone, Copy)]
enum CompressionMode {
    None,
    Lz4,
    ZstdLevel1,
}

impl CompressionMode {
    const fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Lz4 => "lz4",
            Self::ZstdLevel1 => "zstd_level1",
        }
    }

    const fn kind(self) -> Option<ValuePayloadCompressionKind> {
        match self {
            Self::None => None,
            Self::Lz4 => Some(ValuePayloadCompressionKind::Lz4),
            Self::ZstdLevel1 => Some(ValuePayloadCompressionKind::ZstdLevel1),
        }
    }
}

pub(crate) fn workload(c: &mut Criterion) {
    let profile = ValueProfile::Large;
    let datasets = [
        (
            ValueEntropy::Random,
            build_compression_dataset(RECORD_COUNT, VALUE_LEN, VALUE_JITTER, ValueEntropy::Random),
        ),
        (
            ValueEntropy::TemplateNoise,
            build_compression_dataset(
                RECORD_COUNT,
                VALUE_LEN,
                VALUE_JITTER,
                ValueEntropy::TemplateNoise,
            ),
        ),
        (
            ValueEntropy::CorrelatedSeries,
            build_compression_dataset(
                RECORD_COUNT,
                VALUE_LEN,
                VALUE_JITTER,
                ValueEntropy::CorrelatedSeries,
            ),
        ),
        (
            ValueEntropy::RepeatedRuns,
            build_compression_dataset(
                RECORD_COUNT,
                VALUE_LEN,
                VALUE_JITTER,
                ValueEntropy::RepeatedRuns,
            ),
        ),
    ];

    report_space(profile, &datasets);

    for (entropy, dataset) in &datasets {
        bench_ordered_fetch(c, profile, *entropy, dataset);
        bench_iter_all(c, profile, *entropy, dataset);
        bench_append_publish(c, profile, *entropy, dataset);
    }
}

fn bench_ordered_fetch(
    c: &mut Criterion,
    profile: ValueProfile,
    entropy: ValueEntropy,
    dataset: &Dataset,
) {
    let mut group = c.benchmark_group(format!(
        "{}/{}/compression_ordered_fetch",
        WORKLOAD_NAME,
        entropy.name()
    ));
    group.throughput(Throughput::Elements(dataset.ordered_keys.len() as u64));

    for mode in compression_modes() {
        let tempdir = tempfile::tempdir().expect("tempdir should work");
        let store = create_filled_store(
            tempdir.path(),
            profile,
            mode,
            &dataset.entries,
            &commit_options(),
        );
        group.bench_function(mode.name(), |b| {
            b.iter(|| black_box(sum_segment_fetches(&store, &dataset.ordered_keys)))
        });
    }

    group.finish();
}

fn bench_iter_all(
    c: &mut Criterion,
    profile: ValueProfile,
    entropy: ValueEntropy,
    dataset: &Dataset,
) {
    let mut group = c.benchmark_group(format!(
        "{}/{}/compression_iter_all",
        WORKLOAD_NAME,
        entropy.name()
    ));
    group.throughput(Throughput::Elements(dataset.entries.len() as u64));

    for mode in compression_modes() {
        let tempdir = tempfile::tempdir().expect("tempdir should work");
        let store = create_filled_store(
            tempdir.path(),
            profile,
            mode,
            &dataset.entries,
            &commit_options(),
        );
        group.bench_function(mode.name(), |b| {
            b.iter(|| black_box(sum_segment_iter(&store)))
        });
    }

    group.finish();
}

fn bench_append_publish(
    c: &mut Criterion,
    profile: ValueProfile,
    entropy: ValueEntropy,
    dataset: &Dataset,
) {
    let mut group = c.benchmark_group(format!(
        "{}/{}/compression_append_publish",
        WORKLOAD_NAME,
        entropy.name()
    ));
    group.throughput(Throughput::Elements(dataset.entries.len() as u64));

    for mode in compression_modes() {
        group.bench_function(mode.name(), |b| {
            b.iter_batched(
                || tempfile::tempdir().expect("tempdir should work"),
                |tempdir| {
                    let store = create_segment_store(tempdir.path(), profile, mode);
                    fill_segment_store_with_options(&store, &dataset.entries, &commit_options());
                },
                BatchSize::LargeInput,
            )
        });
    }

    group.finish();
}

fn report_space(profile: ValueProfile, datasets: &[(ValueEntropy, Dataset)]) {
    for (entropy, dataset) in datasets {
        let raw_value_bytes = dataset
            .entries
            .iter()
            .map(|(_, value)| value.len() as u64)
            .sum::<u64>();
        for mode in compression_modes() {
            let tempdir = tempfile::tempdir().expect("tempdir should work");
            let _store = create_filled_store(
                tempdir.path(),
                profile,
                mode,
                &dataset.entries,
                &commit_options(),
            );
            let store_bytes = dir_size(tempdir.path());
            eprintln!(
                "{}/{}/compression_space mode={} block={}k raw_value_bytes={} store_bytes={} space_amp={:.3}",
                WORKLOAD_NAME,
                entropy.name(),
                mode.name(),
                DEFAULT_BLOCK_SIZE / 1024,
                raw_value_bytes,
                store_bytes,
                store_bytes as f64 / raw_value_bytes as f64,
            );
        }
    }
}

fn create_filled_store(
    root: &Path,
    profile: ValueProfile,
    mode: CompressionMode,
    entries: &[(Vec<u8>, Vec<u8>)],
    options: &CommitOptions,
) -> Store {
    let store = create_segment_store(root, profile, mode);
    fill_segment_store_with_options(&store, entries, options);
    store
}

fn create_segment_store(root: &Path, profile: ValueProfile, mode: CompressionMode) -> Store {
    let mut options = profile_store_create_options(profile);
    if let Some(kind) = mode.kind() {
        options = options.with_value_payload_compression(kind);
    }
    Store::create(root, options).expect("segment store should create")
}

fn compression_modes() -> [CompressionMode; 3] {
    [
        CompressionMode::None,
        CompressionMode::Lz4,
        CompressionMode::ZstdLevel1,
    ]
}

fn commit_options() -> CommitOptions {
    commit_options_with_block_size(DEFAULT_BLOCK_SIZE)
}

fn dir_size(path: &Path) -> u64 {
    let metadata = std::fs::metadata(path).expect("metadata should read");
    if metadata.is_file() {
        return metadata.len();
    }

    std::fs::read_dir(path)
        .expect("directory should read")
        .map(|entry| {
            let entry = entry.expect("directory entry should read");
            dir_size(&entry.path())
        })
        .sum()
}
