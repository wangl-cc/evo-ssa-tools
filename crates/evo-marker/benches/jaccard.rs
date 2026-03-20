use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use evo_marker::{prelude::*, testutils};
use rand::{SeedableRng, rngs::SmallRng};

const BLOCK: u32 = 12;
const N_BINS: usize = 100;
const MAX_N: usize = 100_000;
const SAMPLE: usize = 10_000;

fn setup_tree() -> LineageTree<BLOCK> {
    let mut rng = SmallRng::seed_from_u64(42);
    let cells = testutils::birth_death_ssa_conditioned(1.0, 0.1, MAX_N, MAX_N, &mut rng);
    testutils::build_tree_sampled(cells, 10.0, SAMPLE, &mut rng)
}

fn bench_jaccard(c: &mut Criterion) {
    let mut group = c.benchmark_group("jaccard");
    let tree = setup_tree();

    group.bench_function("naive", |b| {
        b.iter_batched(
            || tree.clone(),
            |tree| tree.jaccard_distance_dist_leaves::<u32>(N_BINS),
            BatchSize::LargeInput,
        )
    });
    group.bench_function("postorder", |b| {
        b.iter_batched(
            || tree.clone(),
            |tree| tree.jaccard_distance_dist_postorder::<u32>(N_BINS),
            BatchSize::LargeInput,
        )
    });

    group.finish();
}

criterion_group!(benches, bench_jaccard);
criterion_main!(benches);
