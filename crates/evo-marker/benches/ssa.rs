use divan::black_box;
use evo_marker::prelude::*;
use rand::prelude::*;

fn max_size() -> usize {
    black_box(100000)
}

pub fn sample_size() -> usize {
    black_box(10000)
}

fn b() -> f64 {
    black_box(1.0)
}

fn d() -> f64 {
    black_box(0.1)
}

fn lambda() -> f64 {
    black_box(10.0)
}

#[allow(dead_code)]
#[inline]
pub fn pure_birth<M: Marker>(rng: &mut impl Rng) -> Vec<M> {
    birth_death_ssa(b(), d(), max_size(), rng)
}

#[inline]
pub fn birth_death<M: Marker>(rng: &mut impl Rng) -> Vec<M> {
    birth_death_ssa(b(), d(), max_size(), rng)
}

pub fn build_tree<const N: u32>(cells: Vec<LineageNode>, rng: &mut impl Rng) -> PhyloTree<N> {
    PhyloTree::poisson_builder(cells, lambda())
        .unwrap()
        .sample(rng, sample_size())
        .build(rng)
}

pub fn birth_death_ssa<M: Marker>(b: f64, d: f64, max_n: usize, rng: &mut impl Rng) -> Vec<M> {
    let mut cells = Vec::with_capacity(max_n);
    let mut state = M::State::default();
    cells.push(M::default());

    while !cells.is_empty() && cells.len() < max_n {
        let n = cells.len() as f64;
        let birth_propensity = n * b;
        let death_propensity = n * d;
        let total_propensity = birth_propensity + death_propensity;

        let r = rng.random::<f64>() * total_propensity;
        if r < birth_propensity {
            let index = find_n(r, b);
            divide_at(&mut cells, index, &mut state);
        } else {
            let index = find_n(r - birth_propensity, d);
            cells.swap_remove(index);
        }
    }

    cells
}

/// Find the first n such that `(n + 1) * unit >= r`.
///
/// This is useful to find the index of the reaction that happens in a homogeneous reaction system.
fn find_n(r: f64, unit: f64) -> usize {
    (r / unit).ceil() as usize - 1
}
