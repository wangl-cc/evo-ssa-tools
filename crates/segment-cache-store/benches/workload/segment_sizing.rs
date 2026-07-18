use std::{hint::black_box, num::NonZeroUsize};

use criterion::{BatchSize, Criterion, Throughput};
use segment_cache_store::{
    BlockChecksumKind, CommitOptions, CommitStats, CreateOptions, OpenOptions, Store,
    StoreMetadata, StoreStorageStats, WriteBatch,
};

const SPARSE_STRIDE: usize = 16;

const RECORD_SWEEP_PROFILE: KvProfile = KvProfile::new("key_128_value_64", 128, 64, 131_072);

const RECORD_SWEEP_POLICIES: [SegmentPolicy; 5] = [
    SegmentPolicy::new("records_4k_bytes_8m", 4_096, 8 * 1024 * 1024),
    SegmentPolicy::new("records_16k_bytes_16m", 16_384, 16 * 1024 * 1024),
    SegmentPolicy::new("records_32k_bytes_32m", 32_768, 32 * 1024 * 1024),
    SegmentPolicy::new("records_64k_bytes_32m", 65_536, 32 * 1024 * 1024),
    SegmentPolicy::new("records_128k_bytes_32m", 131_072, 32 * 1024 * 1024),
];

const PROFILE_SWEEP_POLICIES: [SegmentPolicy; 4] = [
    SegmentPolicy::new("records_4k_bytes_8m", 4_096, 8 * 1024 * 1024),
    SegmentPolicy::new("records_16k_bytes_16m", 16_384, 16 * 1024 * 1024),
    SegmentPolicy::new("records_32k_bytes_16m", 32_768, 16 * 1024 * 1024),
    SegmentPolicy::new("records_32k_bytes_32m", 32_768, 32 * 1024 * 1024),
];

const PROFILES: [KvProfile; 4] = [
    KvProfile::new("key_16_value_16", 16, 16, 131_072),
    KvProfile::new("key_512_value_64", 512, 64, 65_536),
    KvProfile::new("key_128_value_1k", 128, 1024, 32_768),
    KvProfile::new("key_128_value_16k", 128, 16 * 1024, 2_048),
];

#[derive(Clone, Copy)]
struct SegmentPolicy {
    name: &'static str,
    max_records: usize,
    max_bytes: usize,
}

#[derive(Clone, Copy)]
struct KvProfile {
    name: &'static str,
    key_len: usize,
    value_len: usize,
    record_count: usize,
}

#[derive(Clone, Copy)]
enum BenchmarkScope {
    RecordSweep,
    Profile(&'static str),
}

struct SegmentDataset {
    profile: KvProfile,
    entries: Vec<(Vec<u8>, Vec<u8>)>,
    ordered_keys: Vec<Vec<u8>>,
    sparse_keys: Vec<Vec<u8>>,
}

struct StoreFixture {
    store: Store,
    root: tempfile::TempDir,
    stats: StoreStorageStats,
}

struct EmptyStoreFixture {
    store: Store,
    batch: Option<WriteBatch>,
    _tempdir: tempfile::TempDir,
}

struct SegmentSizingBenchmark {
    scope: BenchmarkScope,
    policies: &'static [SegmentPolicy],
    dataset: SegmentDataset,
    fixtures: Vec<(SegmentPolicy, StoreFixture)>,
}

pub(crate) fn workload(c: &mut Criterion) {
    SegmentSizingBenchmark::new(
        BenchmarkScope::RecordSweep,
        RECORD_SWEEP_PROFILE,
        &RECORD_SWEEP_POLICIES,
    )
    .run(c);

    for profile in PROFILES {
        SegmentSizingBenchmark::new(
            BenchmarkScope::Profile(profile.name),
            profile,
            &PROFILE_SWEEP_POLICIES,
        )
        .run(c);
    }
}

impl SegmentPolicy {
    const fn new(name: &'static str, max_records: usize, max_bytes: usize) -> Self {
        Self {
            name,
            max_records,
            max_bytes,
        }
    }

    fn options(self) -> CommitOptions {
        CommitOptions::default()
            .with_flush_threshold_records(
                NonZeroUsize::new(self.max_records).expect("record threshold is non-zero"),
            )
            .with_flush_threshold_bytes(
                NonZeroUsize::new(self.max_bytes).expect("byte threshold is non-zero"),
            )
    }

    fn create_store(self, root: &std::path::Path, profile: KvProfile) -> Store {
        Store::create(
            root,
            CreateOptions::new(profile.key_len, store_metadata(), benchmark_checksum())
                .expect("benchmark geometry is valid"),
        )
        .expect("benchmark store should create")
    }

    fn commit(self, store: &Store, dataset: &SegmentDataset) -> CommitStats {
        self.commit_batch(store, dataset.write_batch())
    }

    fn commit_batch(self, store: &Store, batch: WriteBatch) -> CommitStats {
        store
            .commit_batch_with_options(batch, &self.options())
            .expect("benchmark entries should commit")
    }

    fn normalization_probe(self, dataset: &SegmentDataset) -> CommitStats {
        let tempdir = tempfile::tempdir().expect("tempdir should work");
        let store = self.create_store(tempdir.path(), dataset.profile);
        self.commit(&store, dataset);
        let (key, value) = dataset.middle_insertion();
        let mut batch = WriteBatch::new();
        batch.push(&key, &value);
        store
            .commit_batch_with_options(batch, &self.options().with_patch_direct_record_limit(0))
            .expect("forced normalization should commit")
    }
}

impl KvProfile {
    const fn new(
        name: &'static str,
        key_len: usize,
        value_len: usize,
        record_count: usize,
    ) -> Self {
        Self {
            name,
            key_len,
            value_len,
            record_count,
        }
    }

    fn logical_bytes(self) -> usize {
        self.record_count * (self.key_len + self.value_len)
    }
}

impl BenchmarkScope {
    fn group_name(self, operation: &str) -> String {
        match self {
            Self::RecordSweep => format!("segment_sizing/{operation}"),
            Self::Profile(profile) => {
                format!("segment_size_profiles/{profile}/{operation}")
            }
        }
    }
}

impl SegmentDataset {
    fn new(profile: KvProfile) -> Self {
        assert!(profile.key_len >= size_of::<u64>());
        assert!(profile.value_len >= size_of::<u64>());

        let entries = (0..profile.record_count)
            .map(|index| Self::entry(profile, (index as u64) * 2))
            .collect::<Vec<_>>();
        let ordered_keys = entries.iter().map(|(key, _)| key.clone()).collect();
        let sparse_keys = entries
            .iter()
            .step_by(SPARSE_STRIDE)
            .map(|(key, _)| key.clone())
            .collect();
        Self {
            profile,
            entries,
            ordered_keys,
            sparse_keys,
        }
    }

    fn middle_insertion(&self) -> (Vec<u8>, Vec<u8>) {
        Self::entry(self.profile, self.profile.record_count as u64 + 1)
    }

    fn write_batch(&self) -> WriteBatch {
        let mut batch = WriteBatch::new();
        for (key, value) in &self.entries {
            batch.push(key, value);
        }
        batch
    }

    fn entry(profile: KvProfile, sequence: u64) -> (Vec<u8>, Vec<u8>) {
        let mut key = vec![b'x'; profile.key_len];
        key[profile.key_len - size_of::<u64>()..].copy_from_slice(&sequence.to_be_bytes());
        let mut value = vec![sequence as u8; profile.value_len];
        value[..size_of::<u64>()].copy_from_slice(&sequence.to_le_bytes());
        (key, value)
    }
}

impl StoreFixture {
    fn new(policy: SegmentPolicy, dataset: &SegmentDataset) -> Self {
        let tempdir = tempfile::tempdir().expect("tempdir should work");
        let store = policy.create_store(tempdir.path(), dataset.profile);
        let commit = policy.commit(&store, dataset);
        let stats = store.storage_stats().expect("storage stats should load");
        assert_eq!(commit.segments_published, stats.segment_files);
        Self {
            store,
            root: tempdir,
            stats,
        }
    }
}

impl EmptyStoreFixture {
    fn new(policy: SegmentPolicy, dataset: &SegmentDataset) -> Self {
        let tempdir = tempfile::tempdir().expect("tempdir should work");
        let store = policy.create_store(tempdir.path(), dataset.profile);
        Self {
            store,
            batch: Some(dataset.write_batch()),
            _tempdir: tempdir,
        }
    }

    fn commit(&mut self, policy: SegmentPolicy) -> CommitStats {
        let batch = self
            .batch
            .take()
            .expect("criterion invokes each prepared input once");
        policy.commit_batch(&self.store, batch)
    }
}

impl SegmentSizingBenchmark {
    fn new(scope: BenchmarkScope, profile: KvProfile, policies: &'static [SegmentPolicy]) -> Self {
        let dataset = SegmentDataset::new(profile);
        let fixtures = policies
            .iter()
            .copied()
            .map(|policy| (policy, StoreFixture::new(policy, &dataset)))
            .collect();
        Self {
            scope,
            policies,
            dataset,
            fixtures,
        }
    }

    fn run(&self, c: &mut Criterion) {
        self.report_layout();
        self.report_normalization_amplification();
        self.bench_open(c);
        self.bench_ordered_fetch(c);
        self.bench_strided_sparse_fetch(c);
        self.bench_append_publish(c);
    }

    fn report_layout(&self) {
        for (policy, fixture) in &self.fixtures {
            let average_segment_bytes = fixture.stats.segment_bytes
                / u64::try_from(fixture.stats.segment_files).expect("segment count fits u64");
            match self.scope {
                BenchmarkScope::RecordSweep => eprintln!(
                    "segment_sizing/layout policy={} segment_files={} segment_bytes={} average_segment_bytes={}",
                    policy.name,
                    fixture.stats.segment_files,
                    fixture.stats.segment_bytes,
                    average_segment_bytes,
                ),
                BenchmarkScope::Profile(profile) => eprintln!(
                    "segment_size_profiles/layout profile={} logical_bytes={} policy={} segment_files={} segment_bytes={} average_segment_bytes={}",
                    profile,
                    self.dataset.profile.logical_bytes(),
                    policy.name,
                    fixture.stats.segment_files,
                    fixture.stats.segment_bytes,
                    average_segment_bytes,
                ),
            }
        }
    }

    fn report_normalization_amplification(&self) {
        for &policy in self.policies {
            let stats = policy.normalization_probe(&self.dataset);
            match self.scope {
                BenchmarkScope::RecordSweep => eprintln!(
                    "segment_sizing/normalization policy={} input_records={} output_records={} segments_retired={} segments_published={}",
                    policy.name,
                    stats.input_records,
                    stats.output_records,
                    stats.segments_retired,
                    stats.segments_published,
                ),
                BenchmarkScope::Profile(profile) => eprintln!(
                    "segment_size_profiles/normalization profile={} policy={} input_records={} output_records={} output_bytes={} segments_retired={} segments_published={}",
                    profile,
                    policy.name,
                    stats.input_records,
                    stats.output_records,
                    stats.output_records
                        * (self.dataset.profile.key_len + self.dataset.profile.value_len),
                    stats.segments_retired,
                    stats.segments_published,
                ),
            }
        }
    }

    fn bench_open(&self, c: &mut Criterion) {
        let mut group = c.benchmark_group(self.scope.group_name("open"));
        for (policy, fixture) in &self.fixtures {
            group.bench_function(policy.name, |b| {
                b.iter(|| {
                    black_box(
                        Store::open(
                            fixture.root.path(),
                            OpenOptions::read_only(store_metadata()),
                        )
                        .expect("benchmark store should reopen"),
                    )
                })
            });
        }
        group.finish();
    }

    fn bench_ordered_fetch(&self, c: &mut Criterion) {
        let mut group = c.benchmark_group(self.scope.group_name("ordered_fetch"));
        group.throughput(Throughput::Elements(self.dataset.ordered_keys.len() as u64));
        for (policy, fixture) in &self.fixtures {
            group.bench_function(policy.name, |b| {
                b.iter(|| {
                    black_box(Self::visit_checksum(
                        &fixture.store,
                        &self.dataset.ordered_keys,
                    ))
                })
            });
        }
        group.finish();
    }

    fn bench_strided_sparse_fetch(&self, c: &mut Criterion) {
        let mut group = c.benchmark_group(self.scope.group_name("strided_sparse_ordered_fetch"));
        group.throughput(Throughput::Elements(self.dataset.sparse_keys.len() as u64));
        for (policy, fixture) in &self.fixtures {
            group.bench_function(policy.name, |b| {
                b.iter(|| {
                    black_box(Self::visit_checksum(
                        &fixture.store,
                        &self.dataset.sparse_keys,
                    ))
                })
            });
        }
        group.finish();
    }

    fn bench_append_publish(&self, c: &mut Criterion) {
        let mut group = c.benchmark_group(self.scope.group_name("append_publish"));
        group.throughput(Throughput::Elements(self.dataset.entries.len() as u64));
        for &policy in self.policies {
            group.bench_function(policy.name, |b| {
                b.iter_batched_ref(
                    || EmptyStoreFixture::new(policy, &self.dataset),
                    |fixture| black_box(fixture.commit(policy).segments_published),
                    BatchSize::LargeInput,
                )
            });
        }
        group.finish();
    }

    fn visit_checksum(store: &Store, keys: &[Vec<u8>]) -> usize {
        let mut checksum = 0usize;
        store
            .visit_many_ordered(keys, |_, value| {
                if let Some(value) = value {
                    checksum = checksum.wrapping_add(value.len());
                }
            })
            .expect("ordered lookup should succeed");
        checksum
    }
}

fn store_metadata() -> StoreMetadata {
    StoreMetadata::from_text("segment-cache-store-segment-sizing-bench")
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
