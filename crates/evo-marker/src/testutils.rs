//! Shared test fixtures for unit tests and benchmarks.
//!
//! Gated behind the `testutils` feature flag. Benchmarks enable it via
//! `required-features = ["testutils"]` in Cargo.toml; unit tests get it
//! via `#[cfg(test)]`.

use rand::{Rng, RngExt};

use crate::prelude::*;

/// Simulate a birth-death process, returning the surviving cells.
///
/// - `b` — per-cell birth rate
/// - `d` — per-cell death rate
/// - `max_n` — stop when the population reaches this size
pub fn birth_death_ssa(b: f64, d: f64, max_n: usize, rng: &mut impl Rng) -> Vec<LineageNode> {
    let mut cells: Vec<LineageNode> = Vec::with_capacity(max_n);
    cells.push(LineageNode::default());

    while !cells.is_empty() && cells.len() < max_n {
        let n = cells.len();
        let r: f64 = rng.random::<f64>() * (b + d);
        if r < b || d == 0.0 {
            let index = rng.random_range(0..n);
            divide_at(&mut cells, index, &mut ());
        } else {
            let index = rng.random_range(0..n);
            cells.swap_remove(index);
        }
    }

    cells
}

/// Simulate a birth-death process until at least `min_survivors` cells survive.
///
/// This is an explicitly conditioned fixture helper for benchmarks and randomized tests that need
/// a minimum number of leaves to build a tree or draw a sample.
#[cfg_attr(coverage_nightly, coverage(off))]
pub fn birth_death_ssa_conditioned(
    b: f64,
    d: f64,
    max_n: usize,
    min_survivors: usize,
    rng: &mut impl Rng,
) -> Vec<LineageNode> {
    assert!(
        min_survivors <= max_n,
        "birth_death_ssa_conditioned requires min_survivors <= max_n"
    );
    assert!(
        min_survivors == 0 || b > 0.0,
        "birth_death_ssa_conditioned requires a positive birth rate when min_survivors > 0"
    );

    loop {
        let cells = birth_death_ssa(b, d, max_n, rng);
        if cells.len() >= min_survivors {
            return cells;
        }
    }
}

/// Build a [`LineageTree`] from cells with Poisson-distributed mutations.
#[cfg_attr(coverage_nightly, coverage(off))]
pub fn build_tree<const N: u32>(
    cells: Vec<LineageNode>,
    lambda: f64,
    rng: &mut impl Rng,
) -> LineageTree<N> {
    LineageTree::poisson_builder(cells, lambda)
        .unwrap()
        .build(rng)
}

/// Build a [`LineageTree`] from cells with Poisson-distributed mutations,
/// sampling `n_sample` leaves.
#[cfg_attr(coverage_nightly, coverage(off))]
pub fn build_tree_sampled<const N: u32>(
    cells: Vec<LineageNode>,
    lambda: f64,
    n_sample: usize,
    rng: &mut impl Rng,
) -> LineageTree<N> {
    LineageTree::poisson_builder(cells, lambda)
        .unwrap()
        .sample(rng, n_sample)
        .build(rng)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use rand::{SeedableRng, rngs::SmallRng};

    use super::*;

    #[test]
    fn test_birth_death_ssa_allows_extinction() {
        let mut rng = SmallRng::seed_from_u64(0);
        let cells = birth_death_ssa(0.0, 1.0, 8, &mut rng);
        assert!(cells.is_empty());
    }

    #[test]
    fn test_birth_death_ssa_conditioned_meets_threshold() {
        let mut rng = SmallRng::seed_from_u64(0);
        let cells = birth_death_ssa_conditioned(1.0, 0.5, 8, 4, &mut rng);
        assert!(cells.len() >= 4);
    }
}
