#![allow(dead_code)]

use std::time::Duration;

use criterion::{BatchSize, BenchmarkId, Criterion};
use evo_marker::{prelude::*, testutils};
use rand::{SeedableRng, rngs::SmallRng};

const MAX_N: usize = 100_000;
const SAMPLE: usize = 10_000;
const RNG_SEED: u64 = 42;

fn setup_rng() -> SmallRng {
    SmallRng::seed_from_u64(RNG_SEED)
}

fn setup_cells() -> (Vec<LineageNode>, SmallRng) {
    let mut rng = setup_rng();
    let cells = testutils::birth_death_ssa_conditioned(1.0, 0.1, MAX_N, MAX_N, &mut rng);
    (cells, rng)
}

pub fn bench_simulate(c: &mut Criterion, allocator: &'static str) {
    let mut group = c.benchmark_group("lineage_alloc_cmp");
    group.sample_size(40);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(4));
    group.bench_function(
        BenchmarkId::new("birth_death_ssa_conditioned", allocator),
        |b| {
            b.iter(|| {
                let mut rng = setup_rng();
                testutils::birth_death_ssa_conditioned(1.0, 0.1, MAX_N, MAX_N, &mut rng)
            })
        },
    );
    group.finish();
}

pub fn bench_build(c: &mut Criterion, allocator: &'static str) {
    let mut group = c.benchmark_group("build_tree_alloc_cmp");
    group.sample_size(60);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(4));
    group.bench_function(BenchmarkId::new("build_tree_sampled", allocator), |b| {
        // `iter_batched` excludes `setup_cells()` from the timed region, so this isolates tree
        // build.
        b.iter_batched(
            setup_cells,
            |(cells, mut rng)| testutils::build_tree_sampled::<12>(cells, 10.0, SAMPLE, &mut rng),
            BatchSize::LargeInput,
        )
    });
    group.finish();
}

pub fn bench_all(c: &mut Criterion, allocator: &'static str) {
    bench_simulate(c, allocator);
    bench_build(c, allocator);
}
