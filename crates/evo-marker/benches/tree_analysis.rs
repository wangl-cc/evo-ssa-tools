use divan::Bencher;
use evo_marker::prelude::*;
use rand::{SeedableRng, rngs::SmallRng};
use ssa::*;

mod ssa;

fn main() {
    divan::main();
}

const BLOCK_SIZES: [u32; 3] = [8, 12, 16];

fn setup_rng() -> SmallRng {
    SmallRng::seed_from_u64(42)
}

fn setup_cells() -> (Vec<LineageNode>, SmallRng) {
    (birth_death(&mut setup_rng()), setup_rng())
}

fn setup_tree<const N: u32>() -> PhyloTree<N> {
    let (cells, mut rng) = setup_cells();
    build_tree::<N>(cells, &mut rng)
}

// #[divan::bench(types = [NoMarker, LineageNode])]
// fn markers<M: Marker>(b: Bencher) {
//     b.with_inputs(setup_rng)
//         .bench_local_refs(|rng| birth_death::<M>(rng));
// }

#[divan::bench(consts = BLOCK_SIZES)]
fn build_phylo_tree<const N: u32>(b: Bencher) {
    b.with_inputs(setup_cells)
        .bench_local_values(|(cells, mut rng)| build_tree::<N>(cells, &mut rng));
}

#[divan::bench(consts = BLOCK_SIZES)]
fn sfs<const N: u32>(b: Bencher) {
    let tree = setup_tree::<N>();
    b.with_inputs(|| tree.clone())
        .bench_local_refs(|tree| tree.sfs());
}

#[divan::bench(consts = BLOCK_SIZES)]
fn mbd<const N: u32>(b: Bencher) {
    let tree = setup_tree::<N>();
    b.with_inputs(|| tree.clone())
        .bench_local_refs(|tree| tree.mbd());
}

#[divan::bench(consts = BLOCK_SIZES)]
fn distance_distribution<const N: u32>(b: Bencher) {
    let tree = setup_tree::<N>();
    b.with_inputs(|| tree.clone())
        .bench_local_refs(|tree| tree.distance_dist_leaves::<u32>());
}
