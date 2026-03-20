// Alternative algorithms explored during development (not implemented):
//
// - **DFS with RLE**: Single tree traversal, no LCA queries. At each internal node `v`, cross-pairs
//   share `LCA = v`, so distance = `(depth_i − nm_v) + (depth_j − nm_v)`. Maintains sorted depth
//   lists per subtree; run-length encoding batches equal depths for O(nnz_left × nnz_right) per
//   node. Useful when per-leaf depths (not just histograms) are needed, e.g. for distance
//   quantiles.
//
// - **Centroid decomposition**: Divide-and-conquer via centroid finding. At each level: find
//   centroid `c` by greedy descent (step into child with >n/2 leaves), compute `d(leaf, c)` for all
//   leaves, count cross-branch pairs, recurse on each branch. O(N log²N) LCA queries. Useful as a
//   general framework for tree path queries (k-nearest, threshold filtering, weighted metrics)
//   where the decomposition structure itself — not just the counting — is needed.

use std::ops::Range;

use frequency::prelude::*;

use super::LineageTree;

// ─── Naive O(N²) with LCA queries ───────────────────────────────────────────

impl<const N: u32> LineageTree<N> {
    /// Iterate over all pairs in `nodes`, yielding `(nm_i, nm_j, nm_lca)`.
    fn pairwise_mutations(
        &self,
        nodes: Range<usize>,
    ) -> impl Iterator<Item = (u16, u16, u16)> + '_ {
        let end = nodes.end;
        (nodes.start..(end - 1)).flat_map(move |node_i| {
            let nm_i = self.total_mutations[node_i];
            ((node_i + 1)..end).map(move |node_j| {
                let nm_j = self.total_mutations[node_j];
                let lca = self.lca_query(node_i, node_j);
                let nm_lca = self.total_mutations[lca];
                (nm_i, nm_j, nm_lca)
            })
        })
    }

    /// Calculate the pairwise distance distribution between `nodes`
    fn distance_dist_naive<T: Count + Sync + Send>(&self, nodes: Range<usize>) -> Vec<T> {
        self.pairwise_mutations(nodes)
            .map(|(nm_i, nm_j, nm_lca)| nm_i + nm_j - 2 * nm_lca)
            .into_bounded_iter(2 * self.max_n_mutations as usize)
            .freq()
    }

    /// Calculate the pairwise distance distribution between all leaves
    pub fn distance_dist_leaves<T: Count + Sync + Send>(&self) -> Vec<T> {
        self.distance_dist_naive(1..(self.n_leaves + 1))
    }

    /// Histogram of pairwise Jaccard distances between `nodes`.
    ///
    /// Jaccard distance = |symmetric difference| / |union|
    ///                  = (nm_i + nm_j - 2 * nm_lca) / (nm_i + nm_j - nm_lca)
    ///
    /// Values lie in \[0, 1\] and are binned into `n_bins` equal-width buckets.
    /// Memory usage is O(`n_bins`), regardless of the number of pairs.
    fn jaccard_distance_dist<T: Count>(&self, nodes: Range<usize>, n_bins: usize) -> Vec<T> {
        self.pairwise_mutations(nodes)
            .map(|(nm_i, nm_j, nm_lca)| {
                let (nm_i, nm_j, nm_lca) = (nm_i as f32, nm_j as f32, nm_lca as f32);
                let union = nm_i + nm_j - nm_lca;
                if union == 0.0 {
                    0.0f32
                } else {
                    (nm_i + nm_j - 2.0 * nm_lca) / union
                }
            })
            .into_binned_iter(n_bins, 0.0f32, 1.0f32)
            .freq()
    }

    /// Histogram of pairwise Jaccard distances between all leaves
    pub fn jaccard_distance_dist_leaves<T: Count>(&self, n_bins: usize) -> Vec<T> {
        self.jaccard_distance_dist(1..(self.n_leaves + 1), n_bins)
    }
}

// ─── Postorder DP with histogram convolution ────────────────────────────────
//
// Both standard distance and Jaccard distance share the same recursive
// tree walk and histogram merge. Only the convolution step differs:
//
// - Standard: freq[d1 + d2 + offset] += cnt1 * cnt2  (contiguous write)
// - Jaccard:  freq[bin(s/(s+nm_v))]   += cnt1 * cnt2  (grouped by bin)

impl<const N: u32> LineageTree<N> {
    /// Postorder DP for leaf pairwise distance distribution.
    ///
    /// Computes the same result as [`distance_dist_leaves`](Self::distance_dist_leaves)
    /// but avoids O(N²) LCA queries by working bottom-up with histogram
    /// convolution.
    ///
    /// # Algorithm
    ///
    /// Each node `v` maintains a histogram `h_v[d]` = number of descendant
    /// leaves at relative mutation depth `d` from `v`. A leaf has `h = [1]`.
    ///
    /// At each internal node `v` with children `c₁, c₂` and edge weights
    /// `w₁ = unique_mutations[c₁]`, `w₂ = unique_mutations[c₂]`:
    ///
    /// 1. **Cross-pair counting** — For every leaf `i` in `c₁`'s subtree and `j` in `c₂`'s subtree,
    ///    the path length is `(d₁ + w₁) + (d₂ + w₂)`:
    ///
    ///    ```text
    ///    freq[d₁ + d₂ + w₁ + w₂] += h₁[d₁] × h₂[d₂]
    ///    ```
    ///
    ///    Implemented as contiguous zip-over-slices writes for
    ///    auto-vectorization.
    ///
    /// 2. **Merge** — Build the parent histogram by shifting each child's histogram by its edge
    ///    weight:
    ///
    ///    ```text
    ///    h_v[d + wₖ] += hₖ[d]   for k ∈ {1, 2}
    ///    ```
    ///
    /// # Complexity
    ///
    /// O(Σ_v |h_left| × |h_right|).
    /// - Balanced tree with Poisson(λ) mutations: O(N · λ²).
    /// - Caterpillar (worst case): O(N²).
    pub fn distance_dist_postorder<T: Count>(&self) -> Vec<T> {
        let max_dist = 2 * self.max_n_mutations as usize;
        let mut freq = vec![0u64; max_dist + 1];
        self.postorder_walk(0, &mut freq, &|h1, h2, w1, w2, _node, freq| {
            let offset = w1 + w2;
            for (d1, &cnt1) in h1.iter().enumerate() {
                if cnt1 == 0 {
                    continue;
                }
                let cnt1 = cnt1 as u64;
                let base = d1 + offset;
                for (f, &cnt2) in freq[base..base + h2.len()].iter_mut().zip(h2.iter()) {
                    *f += cnt1 * cnt2 as u64;
                }
            }
        });

        let mut result: Vec<T> = freq.iter().map(|&c| T::from_count(c as usize)).collect();
        truncate_trailing_zeros(&mut result);
        result
    }

    /// Postorder DP for Jaccard distance histogram.
    ///
    /// Computes the same result as
    /// [`jaccard_distance_dist_leaves`](Self::jaccard_distance_dist_leaves) but uses the same
    /// bottom-up histogram convolution as
    /// [`distance_dist_postorder`](Self::distance_dist_postorder).
    ///
    /// # Algorithm
    ///
    /// At each node `v`, the Jaccard distance for a cross-pair with
    /// relative depth sum `s = d₁ + w₁ + d₂ + w₂` is:
    ///
    /// ```text
    /// J = s / (s + nm_v)
    /// ```
    ///
    /// where `nm_v = total_mutations[v]` (shared mutations cancel out).
    ///
    /// ## Bin-threshold optimization
    ///
    /// The naive convolution would compute `J` for every `(d₁, d₂)` pair
    /// — O(|h₁| × |h₂|) float divisions and scatter writes.
    ///
    /// Key insight: for fixed `d₁`, `J` is monotonically increasing in
    /// `d₂`, so consecutive `d₂` values map to non-decreasing bins. We
    /// precompute the `s`-threshold where each bin boundary falls:
    ///
    /// ```text
    /// J ≥ b/n_bins  ⟺  s ≥ ⌈b · nm_v / (n_bins − b)⌉
    /// ```
    ///
    /// Combined with prefix sums on `h₂`, each bin's count becomes an
    /// O(1) range-sum query. The inner loop changes from O(|h₂|) float
    /// divisions to O(n_bins) integer comparisons per `d₁` value.
    pub fn jaccard_distance_dist_postorder<T: Count>(&self, n_bins: usize) -> Vec<T> {
        let mut freq = vec![0u64; n_bins];
        self.postorder_walk(0, &mut freq, &|h1, h2, w1, w2, node, freq| {
            let nm_v = self.total_mutations[node] as u64;
            let n_bins = freq.len();
            let offset = (w1 + w2) as u64;

            // Prefix sums of h2 for O(1) range-sum queries.
            let h2_prefix = prefix_sums_u64(h2);

            if nm_v == 0 {
                // jaccard = s/s = 1 for s > 0, 0/0 = 0 for s = 0.
                let last = n_bins - 1;
                let total_h2 = *h2_prefix.last().unwrap();
                for (d1, &cnt1) in h1.iter().enumerate() {
                    if cnt1 == 0 {
                        continue;
                    }
                    let cnt1 = cnt1 as u64;
                    if d1 as u64 + offset == 0 && !h2.is_empty() {
                        freq[0] += cnt1 * h2[0] as u64;
                        freq[last] += cnt1 * (total_h2 - h2[0] as u64);
                    } else {
                        freq[last] += cnt1 * total_h2;
                    }
                }
                return;
            }

            // Bin thresholds: bin b starts at s >= ceil(b * nm_v / (n_bins - b)).
            let s_thresh: Vec<u64> = (0..=n_bins)
                .map(|b| {
                    if b == 0 {
                        0
                    } else if b >= n_bins {
                        u64::MAX
                    } else {
                        let numer = b as u64 * nm_v;
                        let denom = (n_bins - b) as u64;
                        numer.div_ceil(denom)
                    }
                })
                .collect();

            let h2_len = h2.len() as u64;
            for (d1, &cnt1) in h1.iter().enumerate() {
                if cnt1 == 0 {
                    continue;
                }
                let cnt1 = cnt1 as u64;
                let s_base = d1 as u64 + offset;

                for b in 0..n_bins {
                    let d2_lo = s_thresh[b].saturating_sub(s_base).min(h2_len);
                    let d2_hi = s_thresh[b + 1].saturating_sub(s_base).min(h2_len);
                    if d2_lo >= d2_hi {
                        continue;
                    }
                    let sum = h2_prefix[d2_hi as usize] - h2_prefix[d2_lo as usize];
                    freq[b] += cnt1 * sum;
                }
            }
        });
        freq.iter().map(|&c| T::from_count(c as usize)).collect()
    }

    /// Shared postorder walk skeleton.
    ///
    /// Recursively walks the tree bottom-up. At each internal node `v`:
    /// 1. Recurse into children to obtain their depth histograms `h₁, h₂`.
    /// 2. Call `convolve(h₁, h₂, w₁, w₂, node, freq)` to count cross-pair contributions (the
    ///    metric-specific step).
    /// 3. Merge `h₁` and `h₂` into a single histogram for `v`'s parent, shifting each by its edge
    ///    weight.
    ///
    /// Returns the depth histogram for `node`'s subtree.
    fn postorder_walk(
        &self,
        node: usize,
        freq: &mut [u64],
        convolve: &impl Fn(&[u32], &[u32], usize, usize, usize, &mut [u64]),
    ) -> Vec<u32> {
        let Some([c1, c2]) = self.children(node) else {
            return vec![1];
        };

        let w1 = self.unique_mutations[c1 as usize] as usize;
        let w2 = self.unique_mutations[c2 as usize] as usize;
        let h1 = self.postorder_walk(c1 as usize, freq, convolve);
        let h2 = self.postorder_walk(c2 as usize, freq, convolve);

        convolve(&h1, &h2, w1, w2, node, freq);

        // Merge: reuse the larger child's buffer to avoid an allocation.
        let size1 = h1.len() + w1;
        let size2 = h2.len() + w2;
        let needed = size1.max(size2);
        let (mut merged, other, w_m, w_o) = if size1 >= size2 {
            (h1, h2, w1, w2)
        } else {
            (h2, h1, w2, w1)
        };
        let old_len = merged.len();
        merged.resize(needed, 0);
        if w_m > 0 {
            merged.copy_within(0..old_len, w_m);
            merged[..w_m].fill(0);
        }
        for (i, &cnt) in other.iter().enumerate() {
            merged[i + w_o] += cnt;
        }
        merged
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Build prefix sums: `result[i] = sum(h[0..i])`, `result[0] = 0`.
fn prefix_sums_u64(h: &[u32]) -> Vec<u64> {
    let mut prefix = Vec::with_capacity(h.len() + 1);
    prefix.push(0u64);
    let mut acc = 0u64;
    for &cnt in h {
        acc += cnt as u64;
        prefix.push(acc);
    }
    prefix
}

/// Remove trailing zeros to match the output format of `.freq()`.
fn truncate_trailing_zeros<T: Count>(freq: &mut Vec<T>) {
    while freq.last() == Some(&T::ZERO) {
        freq.pop();
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use rand::{SeedableRng, rngs::SmallRng};

    use super::super::*;

    #[derive(Debug, Clone, Copy)]
    struct Const(u16);

    impl rand::distr::Distribution<u16> for Const {
        fn sample<R: rand::Rng + ?Sized>(&self, _: &mut R) -> u16 {
            self.0
        }
    }

    fn divide_at(cells: &mut Vec<crate::lineage::node::LineageNode>, i: usize) {
        crate::divide_at(cells, i, &mut ());
    }

    fn rng() -> SmallRng {
        SmallRng::seed_from_u64(0)
    }

    // ── Tree builders ───────────────────────────────────────────────────

    fn six_leaf_tree() -> LineageTree<8> {
        let mut cells = vec![crate::lineage::node::LineageNode::default()];
        let cells_mut_ref = &mut cells;
        divide_at(cells_mut_ref, 0);
        divide_at(cells_mut_ref, 0);
        divide_at(cells_mut_ref, 1);
        divide_at(cells_mut_ref, 0);
        divide_at(cells_mut_ref, 1);
        LineageTree::builder(cells, Const(1)).build(&mut rng())
    }

    fn absorbing_root_tree() -> LineageTree<2> {
        let mut cells = vec![crate::lineage::node::LineageNode::default()];
        let cells_mut_ref = &mut cells;
        divide_at(cells_mut_ref, 0);
        divide_at(cells_mut_ref, 0);
        cells_mut_ref.remove(1);
        LineageTree::builder(cells, Const(1)).build(&mut rng())
    }

    fn asymmetric_tree() -> LineageTree<8> {
        let mut cells = vec![crate::lineage::node::LineageNode::default()];
        let cells_mut_ref = &mut cells;
        divide_at(cells_mut_ref, 0);
        divide_at(cells_mut_ref, 0);
        divide_at(cells_mut_ref, 1);
        divide_at(cells_mut_ref, 0);
        divide_at(cells_mut_ref, 1);
        cells_mut_ref.remove(0);
        LineageTree::builder(cells, Const(1)).build(&mut rng())
    }

    fn right_heavy_tree() -> LineageTree<8> {
        let mut cells = vec![crate::lineage::node::LineageNode::default()];
        let cells_mut_ref = &mut cells;
        divide_at(cells_mut_ref, 0);
        divide_at(cells_mut_ref, 1);
        divide_at(cells_mut_ref, 1);
        divide_at(cells_mut_ref, 2);
        LineageTree::builder(cells, Const(1)).build(&mut rng())
    }

    fn zero_mutation_tree() -> LineageTree<2> {
        let mut cells = vec![crate::lineage::node::LineageNode::default()];
        let cells_mut_ref = &mut cells;
        divide_at(cells_mut_ref, 0);
        divide_at(cells_mut_ref, 0);
        divide_at(cells_mut_ref, 1);
        LineageTree::builder(cells, Const(0)).build(&mut rng())
    }

    // ── Postorder matches naive ─────────────────────────────────────────

    #[test]
    fn test_postorder_matches_naive() {
        for phylo in [six_leaf_tree(), asymmetric_tree(), right_heavy_tree()] {
            let expected: Vec<u32> = phylo.distance_dist_leaves();
            assert_eq!(phylo.distance_dist_postorder::<u32>(), expected);
        }
    }

    #[test]
    fn test_postorder_matches_naive_absorbing_root() {
        let phylo = absorbing_root_tree();
        let expected: Vec<u32> = phylo.distance_dist_leaves();
        assert_eq!(phylo.distance_dist_postorder::<u32>(), expected);
    }

    // ── Jaccard naive ───────────────────────────────────────────────────

    #[test]
    fn test_jaccard_distance_dist_leaves() {
        let phylo = six_leaf_tree();
        let hist: Vec<u32> = phylo.jaccard_distance_dist_leaves(4);
        assert_eq!(hist, vec![0, 0, 2, 13]);
    }

    #[test]
    fn test_jaccard_zero_mutations() {
        let phylo = zero_mutation_tree();
        let hist: Vec<u32> = phylo.jaccard_distance_dist_leaves(10);
        assert_eq!(hist[0], 6); // C(4,2) = 6 pairs, all in bin 0
    }

    // ── Jaccard postorder matches naive ─────────────────────────────────

    #[test]
    fn test_jaccard_postorder_matches_naive() {
        for phylo in [six_leaf_tree(), asymmetric_tree(), right_heavy_tree()] {
            let naive: Vec<u32> = phylo.jaccard_distance_dist_leaves(100);
            let fast: Vec<u32> = phylo.jaccard_distance_dist_postorder(100);
            assert_eq!(naive, fast);
        }
    }

    #[test]
    fn test_jaccard_postorder_absorbing_root() {
        let phylo = absorbing_root_tree();
        let naive: Vec<u32> = phylo.jaccard_distance_dist_leaves(50);
        let fast: Vec<u32> = phylo.jaccard_distance_dist_postorder(50);
        assert_eq!(naive, fast);
    }

    #[test]
    fn test_jaccard_postorder_zero_mutations() {
        let phylo = zero_mutation_tree();
        let naive: Vec<u32> = phylo.jaccard_distance_dist_leaves(10);
        let fast: Vec<u32> = phylo.jaccard_distance_dist_postorder(10);
        assert_eq!(naive, fast);
    }

    // ── Randomized fuzz tests ──────────────────────────────────────────

    fn random_tree(seed: u64, n_leaves: usize, death_rate: f64, lambda: f64) -> LineageTree<16> {
        let mut rng = SmallRng::seed_from_u64(seed);
        let cells = crate::testutils::birth_death_ssa_conditioned(
            1.0, death_rate, n_leaves, n_leaves, &mut rng,
        );
        crate::testutils::build_tree(cells, lambda, &mut rng)
    }

    #[test]
    fn test_fuzz_distance_postorder_matches_naive() {
        for seed in 0..20 {
            for &(n, death, lambda) in &[
                (10, 0.0, 1.0),  // pure birth, low mutation
                (50, 0.3, 5.0),  // birth-death, moderate mutation
                (30, 0.1, 0.5),  // low mutation rate
                (20, 0.5, 10.0), // high death, high mutation
            ] {
                let tree = random_tree(seed, n, death, lambda);
                let naive: Vec<u64> = tree.distance_dist_leaves();
                let fast: Vec<u64> = tree.distance_dist_postorder();
                assert_eq!(
                    naive, fast,
                    "seed={seed}, n={n}, death={death}, lambda={lambda}"
                );
            }
        }
    }

    #[test]
    fn test_fuzz_jaccard_postorder_matches_naive() {
        for seed in 0..20 {
            for &(n, death, lambda, n_bins) in &[
                (10, 0.0, 1.0, 10),
                (50, 0.3, 5.0, 50),
                (30, 0.1, 0.5, 100),
                (20, 0.5, 10.0, 20),
            ] {
                let tree = random_tree(seed, n, death, lambda);
                let naive: Vec<u64> = tree.jaccard_distance_dist_leaves(n_bins);
                let fast: Vec<u64> = tree.jaccard_distance_dist_postorder(n_bins);
                assert_eq!(
                    naive, fast,
                    "seed={seed}, n={n}, death={death}, lambda={lambda}, bins={n_bins}"
                );
            }
        }
    }
}
