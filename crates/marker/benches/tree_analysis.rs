use criterion::{Criterion, criterion_group, criterion_main};
use evo_marker::prelude::*;
use rand::{SeedableRng, rngs::SmallRng};
use ssa::*;

mod ssa;

fn bench_birth_death(c: &mut Criterion) {
    let mut group = c.benchmark_group("SSA");
    let mut rng = SmallRng::seed_from_u64(42);
    group.bench_function("SSA no marker", |b| {
        b.iter(|| birth_death::<NoMarker>(&mut rng))
    });
    let mut rng = SmallRng::seed_from_u64(42);
    group.bench_function("SSA lineage", |b| {
        b.iter(|| birth_death::<LineageNode>(&mut rng))
    });
    group.finish();

    let mut rng = SmallRng::seed_from_u64(42);
    let cells = birth_death(&mut rng);
    let cells = down_sample(cells, &mut rng);

    let mut group = c.benchmark_group("Build Phylogenetic Tree");
    let mut rng = SmallRng::seed_from_u64(42);
    group.bench_function("N = 8", |b| {
        b.iter(|| build_tree::<_, 8>(cells.iter(), &mut rng))
    });
    let mut rng = SmallRng::seed_from_u64(42);
    group.bench_function("N = 16", |b| {
        b.iter(|| build_tree::<_, 16>(cells.iter(), &mut rng))
    });
    group.finish();

    let tree_8: PhyloTree<8> = build_tree(cells.iter(), &mut SmallRng::seed_from_u64(42));
    let tree_16: PhyloTree<16> = build_tree(cells.iter(), &mut SmallRng::seed_from_u64(42));

    let mut group = c.benchmark_group("Calculate SFS");
    group.bench_function("N = 8", |b| b.iter(|| tree_8.sfs()));
    group.bench_function("N = 16", |b| b.iter(|| tree_16.sfs()));
    group.finish();

    let mut group = c.benchmark_group("Calculate MBD");
    group.bench_function("N = 8", |b| b.iter(|| tree_8.mbd()));
    group.bench_function("N = 16", |b| b.iter(|| tree_16.mbd()));
    group.finish();

    let mut group = c.benchmark_group("Calculate Distance Distribution");
    group.bench_function("N = 8", |b| b.iter(|| tree_8.distance_dist()));
    group.bench_function("N = 16", |b| b.iter(|| tree_16.distance_dist()));
    group.finish();
}

criterion_group!(benches, bench_birth_death);

criterion_main!(benches);
