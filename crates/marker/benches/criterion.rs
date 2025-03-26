use criterion::{
    BenchmarkGroup, Criterion, criterion_group, criterion_main, measurement::Measurement,
};
use evo_marker::prelude::*;
use rand::{SeedableRng, rngs::SmallRng};
use ssa::*;

mod ssa;

fn add_analysis_benches<M: Measurement>(group: &mut BenchmarkGroup<M>, cells: Vec<LineageNode>) {
    let mut rng = SmallRng::seed_from_u64(42);
    let rng_ref = &mut rng;

    let tree = build_tree(&cells, rng_ref);

    group.bench_function("Build Phylogenic Tree", |b| {
        b.iter(|| build_tree(&cells, rng_ref))
    });
    group.bench_function("Calculate SFS", |b| b.iter(|| tree.sfs()));
    group.bench_function("Calculate MBD", |b| b.iter(|| tree.mbd()));
}

fn bench_pure_birth(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let rng_ref = &mut rng;

    let cells = pure_birth(rng_ref);

    let mut group = c.benchmark_group("Pure Birth Process");
    group.bench_function("SSA no marker", |b| {
        b.iter(|| pure_birth::<NoMarker>(rng_ref))
    });
    group.bench_function("SSA lineage", |b| {
        b.iter(|| pure_birth::<LineageNode>(rng_ref))
    });

    add_analysis_benches(&mut group, cells);
}

fn bench_birth_death(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let rng_ref = &mut rng;

    let cells = birth_death(rng_ref);

    let mut group = c.benchmark_group("Birth Death Process");
    group.bench_function("SSA no marker", |b| {
        b.iter(|| birth_death::<NoMarker>(rng_ref))
    });
    group.bench_function("SSA lineage", |b| {
        b.iter(|| birth_death::<LineageNode>(rng_ref))
    });
    add_analysis_benches(&mut group, cells);
}

criterion_group!(benches, bench_pure_birth, bench_birth_death);

criterion_main!(benches);
