use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use evo_marker::{prelude::*, testutils};
use rand::{SeedableRng, rngs::SmallRng};

const MAX_N: usize = 100_000;
const SAMPLE: usize = 10_000;

fn setup_rng() -> SmallRng {
    SmallRng::seed_from_u64(42)
}

fn setup_cells() -> (Vec<LineageNode>, SmallRng) {
    let mut rng = setup_rng();
    let cells = testutils::birth_death_ssa_conditioned(1.0, 0.1, MAX_N, MAX_N, &mut rng);
    (cells, rng)
}

fn setup_tree<const N: u32>() -> LineageTree<N> {
    let (cells, mut rng) = setup_cells();
    testutils::build_tree_sampled(cells, 10.0, SAMPLE, &mut rng)
}

fn bench_build_phylo_tree(c: &mut Criterion) {
    let mut group = c.benchmark_group("build_phylo_tree");
    group.bench_function("block_8", |b| {
        b.iter_batched(
            setup_cells,
            |(cells, mut rng)| testutils::build_tree_sampled::<8>(cells, 10.0, SAMPLE, &mut rng),
            BatchSize::LargeInput,
        )
    });
    group.bench_function("block_12", |b| {
        b.iter_batched(
            setup_cells,
            |(cells, mut rng)| testutils::build_tree_sampled::<12>(cells, 10.0, SAMPLE, &mut rng),
            BatchSize::LargeInput,
        )
    });
    group.bench_function("block_16", |b| {
        b.iter_batched(
            setup_cells,
            |(cells, mut rng)| testutils::build_tree_sampled::<16>(cells, 10.0, SAMPLE, &mut rng),
            BatchSize::LargeInput,
        )
    });
    group.finish();
}

fn bench_sfs(c: &mut Criterion) {
    let mut group = c.benchmark_group("sfs");
    let tree8 = setup_tree::<8>();
    let tree12 = setup_tree::<12>();
    let tree16 = setup_tree::<16>();
    group.bench_function("block_8", |b| {
        b.iter_batched(|| tree8.clone(), |tree| tree.sfs(), BatchSize::LargeInput)
    });
    group.bench_function("block_12", |b| {
        b.iter_batched(|| tree12.clone(), |tree| tree.sfs(), BatchSize::LargeInput)
    });
    group.bench_function("block_16", |b| {
        b.iter_batched(|| tree16.clone(), |tree| tree.sfs(), BatchSize::LargeInput)
    });
    group.finish();
}

fn bench_mbd(c: &mut Criterion) {
    let mut group = c.benchmark_group("mbd");
    let tree8 = setup_tree::<8>();
    let tree12 = setup_tree::<12>();
    let tree16 = setup_tree::<16>();
    group.bench_function("block_8", |b| {
        b.iter_batched(|| tree8.clone(), |tree| tree.mbd(), BatchSize::LargeInput)
    });
    group.bench_function("block_12", |b| {
        b.iter_batched(|| tree12.clone(), |tree| tree.mbd(), BatchSize::LargeInput)
    });
    group.bench_function("block_16", |b| {
        b.iter_batched(|| tree16.clone(), |tree| tree.mbd(), BatchSize::LargeInput)
    });
    group.finish();
}

criterion_group!(benches, bench_build_phylo_tree, bench_sfs, bench_mbd);
criterion_main!(benches);
