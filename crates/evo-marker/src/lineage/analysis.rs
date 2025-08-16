use std::{fmt::Debug, num::NonZero, ops::Range};

use frequency::prelude::*;
use rand::{Rng, distr::Distribution};

use super::{
    node::LineageNode,
    rmq::{BlockRMQ, BlockRMQBuilder},
};
use crate::util::{distributions::PoissonKnuth, hashers::NoHashMap};

/// A tree structure represents the cell lineage
///
/// The original mutations are in child-parent relationship by `Rc` pointers, which is a tree
/// structure. `Rc` is not suitable to serialize and deserialize and hard to analyze.
/// Here we use euler tour to represent the tree, which is good for serialization and analysis.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "bitcode", derive(bitcode::Decode, bitcode::Encode))]
pub struct PhyloTree<const N: u32> {
    // Basic information
    n_leaves: usize,
    max_n_mutations: u16,
    // Nodes
    unique_mutations: Vec<u16>, // Number of unique mutations of each node in the tree
    total_mutations: Vec<u16>,  // Number of all mutations of each node in the tree
    // Euler tour
    euler_tour: Vec<u32>,        // Euler tour of the tree
    rmq_table: BlockRMQ<N>,      // Range Minimum Query table, used to find the LCA
    first_occurrences: Vec<u32>, // Position of first occurrence of each node in the Euler tour
    last_occurrences: Vec<u32>,  // Position of last occurrence of each node in the Euler tour
}

pub struct PhyloTreeBuilder<const N: u32, D: Distribution<u16> = PoissonKnuth> {
    cells: Vec<LineageNode>,
    dist: D,
}

impl<const N: u32, D: Distribution<u16>> PhyloTreeBuilder<N, D> {
    pub fn sample(mut self, rng: &mut impl rand::Rng, amount: usize) -> Self {
        let indices = {
            let mut indices = rand::seq::index::sample(rng, self.cells.len(), amount).into_vec();
            indices.sort_unstable();
            indices
        };
        for (i, j) in indices.iter().enumerate() {
            self.cells.swap(i, *j);
        }
        self.cells.truncate(indices.len());
        self
    }

    /// Create a new phylogenetic tree from a list of cells.
    ///
    /// # Panics
    ///
    /// Panics if the iterator does not have an exact size, and if the iterator is empty or has only
    /// one element.
    ///
    /// If you call this function with a downsampled set of cells, please make sure all other cells
    /// (not being chosen) are dropped.
    ///
    /// # Undefined Behavior
    ///
    /// If those cells not belonging to the same lineage, i.e. they have multiple root nodes.
    pub fn build(self, rng: &mut impl Rng) -> PhyloTree<N> {
        let cells = self.cells;
        let dist = self.dist;
        let n_leaves = cells.len();
        // For full binary tree, the number of nodes is 2L - 1
        // where L is the number of leaves, and the number of inner nodes is L - 1
        // Despites the division tree may contain some nodes with only one child due to death,
        // but it can be solved by joining those nodes with its only child.
        let n_nodes = (n_leaves << 1) - 1;

        let mut forward_tree = Vec::with_capacity(n_leaves - 1);
        let mut unique_mutations = Vec::with_capacity(n_nodes);
        let mut total_mutations = Vec::with_capacity(n_nodes);

        // The root node
        forward_tree.push(Children::default());
        unique_mutations.push(0);
        total_mutations.push(0);
        // Reserve space for leaves
        forward_tree.resize(n_leaves + 1, Children::default());
        unique_mutations.resize(n_leaves + 1, 0);
        total_mutations.resize(n_leaves + 1, 0);

        // Index for all inner nodes, zero is reserved for the root node, and 1..=n_leaves are
        // reserved for the leaves, so for inner nodes, we start from n_leaves + 1
        let mut id2index = NoHashMap::default();

        let mut max_n_mutations = 0;

        for (i, cell) in cells.iter().enumerate() {
            let index = i + 1; // The zero is reserved for the root node
            let (pindex, to_join) = resolve_parent(
                cell.parent().unwrap(), // Leaf node always has a parent
                &mut id2index,
                &mut forward_tree,
                &mut unique_mutations,
                &mut total_mutations,
                rng,
                &dist,
            );
            // Leaves must have a parent that is not the root node
            let pindex = pindex.unwrap();

            let n_unique_mutation = dist.sample(rng) + to_join;
            let n_total_mutation = total_mutations[pindex] + n_unique_mutation;

            forward_tree[pindex].add_child(NonZero::new(index as u32).unwrap());
            unique_mutations[index] = n_unique_mutation;
            total_mutations[index] = n_total_mutation;

            if n_total_mutation > max_n_mutations {
                max_n_mutations = n_total_mutation;
            }
        }

        // cells must be dropped here, can not drop inside the loop, which will change ref_count
        drop(cells);
        // No longer needed, drop it to free memory
        drop(id2index);

        /// Resolve parent node, used to build forward tree
        ///
        /// Return the index of the node, number of unique mutations to be observed, and whether the
        /// node has been visited before
        fn resolve_parent<G: rand::Rng, D: Distribution<u16>>(
            node: &LineageNode,
            id2index: &mut NoHashMap<usize, usize>,
            forward_tree: &mut Vec<Children>,
            unique_mutations: &mut Vec<u16>,
            total_mutations: &mut Vec<u16>,
            rng: &mut G,
            dist: &D,
        ) -> (Option<usize>, u16) {
            let id = node.id();

            if let Some(&index) = id2index.get(&id.get()) {
                // Only node with 2 children will be cached in the index
                // as 1 child node will only be traversed once
                return (Some(index), 0);
            }

            let (pindex, n_unique_mutation) = if let Some(parent) = node.parent() {
                let (pindex, to_join) = resolve_parent(
                    parent,
                    id2index,
                    forward_tree,
                    unique_mutations,
                    total_mutations,
                    rng,
                    dist,
                );

                let n_unique_mutation = dist.sample(rng) + to_join;

                (pindex, n_unique_mutation)
            } else {
                // If a node has no children, it is the root node
                (None, 0)
            };

            match node.ref_count() {
                // For node with only 1 child, the unique mutations are joined to the child
                1 => (pindex, n_unique_mutation),
                // For node with two children, the unique mutations is its own unique,
                // mutations should not be joined to children
                2 => {
                    // Observe root if root has only one child
                    let index = if let Some(pindex) = pindex {
                        let index = forward_tree.len();
                        forward_tree[pindex].add_child(NonZero::new(index as u32).unwrap());
                        forward_tree.push(Children::default());
                        unique_mutations.push(n_unique_mutation);
                        total_mutations.push(n_unique_mutation + total_mutations[pindex]);
                        index
                    } else {
                        unique_mutations[0] = n_unique_mutation;
                        total_mutations[0] = n_unique_mutation;
                        0
                    };
                    id2index.insert(id.get(), index);
                    // Unwrap: index is always non-zero, as the tree is not empty
                    (Some(index), 0)
                }
                _ => unreachable!("Inner node only possible with 1 or 2 children"),
            }
        }

        assert_eq!(forward_tree.len(), n_nodes);
        assert_eq!(unique_mutations.len(), n_nodes);
        assert_eq!(total_mutations.len(), n_nodes);

        let len_euler_tour = n_nodes * 2 - 1;

        let mut euler_tour = Vec::with_capacity(len_euler_tour);
        let mut rmq_builder = BlockRMQBuilder::with_capacity(len_euler_tour);
        let mut first_occurrences = vec![0; n_nodes];
        let mut last_occurrences = vec![0; n_nodes];

        dfs(
            0,
            &forward_tree,
            &mut euler_tour,
            &mut rmq_builder,
            &mut first_occurrences,
            &mut last_occurrences,
        );

        /// Depth-first search to build the Euler tour and rmq table
        fn dfs<const N: u32>(
            node: u32,
            forward_tree: &[Children],
            euler_tour: &mut Vec<u32>,
            rmq_builder: &mut BlockRMQBuilder<N>,
            first_occurrences: &mut Vec<u32>,
            last_occurrences: &mut Vec<u32>,
        ) {
            euler_tour.push(node);
            first_occurrences[node as usize] = euler_tour.len() as u32 - 1;

            for child in forward_tree[node as usize].iter() {
                rmq_builder.step(false);
                dfs(
                    child.get(),
                    forward_tree,
                    euler_tour,
                    rmq_builder,
                    first_occurrences,
                    last_occurrences,
                );
                euler_tour.push(node);
                rmq_builder.step(true);
            }
            last_occurrences[node as usize] = euler_tour.len() as u32 - 1;
        }

        PhyloTree {
            n_leaves,
            max_n_mutations,
            unique_mutations,
            total_mutations,
            euler_tour,
            rmq_table: rmq_builder.finish(),
            first_occurrences,
            last_occurrences,
        }
    }
}

impl<const N: u32> PhyloTree<N> {
    pub fn builder<D: Distribution<u16>>(
        cells: Vec<LineageNode>,
        dist: D,
    ) -> PhyloTreeBuilder<N, D> {
        PhyloTreeBuilder { cells, dist }
    }

    pub fn poisson_builder(
        cells: Vec<LineageNode>,
        lambda: f64,
    ) -> Option<PhyloTreeBuilder<N, PoissonKnuth>> {
        Some(Self::builder(cells, PoissonKnuth::new(lambda)?))
    }
}

// Util methods
impl<const N: u32> PhyloTree<N> {
    /// Get the number of leaves in a subtree rooted at the given node.
    ///
    /// This is based on one fact that for a full binary tree, the length of the Euler tour `E`
    /// must be `4L - 3`, where `L` is the number of leaves in the tree.
    pub fn n_leaves_subtree(&self, i: usize) -> u32 {
        let first_occurrence = self.first_occurrences[i];
        let last_occurrence = self.last_occurrences[i];
        (last_occurrence - first_occurrence + 4) >> 2
    }

    /// Get two children of a node if exists.
    ///
    /// For a full binary tree, the children of a node must be occurrences after entering the node,
    /// and before leaving the node.
    pub fn children(&self, i: usize) -> Option<[u32; 2]> {
        let fi = self.first_occurrences[i];
        let ei = self.last_occurrences[i];

        if fi == ei {
            None
        } else {
            let child1 = self.euler_tour[fi as usize + 1];
            let child2 = self.euler_tour[ei as usize - 1];
            Some([child1, child2])
        }
    }

    /// Find the least common ancestor of two nodes
    ///
    /// This function uses the RMQ table to find the LCA of two nodes in O(1) time.
    pub fn lca_query(&self, i: usize, j: usize) -> usize {
        let i_first_occurrence = self.first_occurrences[i];
        let j_first_occurrence = self.first_occurrences[j];

        let (start, end) = if i_first_occurrence <= j_first_occurrence {
            (i_first_occurrence, j_first_occurrence)
        } else {
            (j_first_occurrence, i_first_occurrence)
        };
        let lca_in_tour = self.rmq_table.min_in(start, end);
        self.euler_tour[lca_in_tour as usize] as usize
    }
}

impl<const N: u32> PhyloTree<N> {
    /// Calculate the site frequency spectrum of the tree
    pub fn sfs(&self) -> Vec<u32> {
        self.unique_mutations
            .iter()
            .enumerate()
            .map(|(i, &nm)| (self.n_leaves_subtree(i), nm as u32))
            .into_bounded_iter(self.n_leaves)
            .weighted_freq()
    }

    /// Calculate single cell mutation burden distribution (scMBD) for all leaves
    pub fn mbd(&self) -> Vec<u16> {
        self.total_mutations[1..self.n_leaves + 1]
            .iter()
            .copied()
            .into_bounded_iter(self.max_n_mutations as usize)
            .freq()
    }

    /// Calculate unique mutation burden distribution (uMBD) for all leaves
    pub fn umbd(&self) -> Vec<u16> {
        self.unique_mutations
            .iter()
            .copied()
            .into_bounded_iter(self.max_n_mutations as usize)
            .freq()
    }

    /// Calculate balance values of all inner nodes grouped by their unique mutation
    pub fn bbm(&self) -> (Vec<f64>, Vec<f64>) {
        let mut balances_sum = vec![0; self.max_n_mutations as usize + 1];
        let mut norm_balances_sum = vec![0.0; self.max_n_mutations as usize + 1];
        let mut repetitions = vec![0u32; self.max_n_mutations as usize + 1];

        for (node, &nm) in self.total_mutations.iter().enumerate() {
            if let Some(children) = self.children(node) {
                let l_leaves = self.n_leaves_subtree(children[0] as usize);
                let r_leaves = self.n_leaves_subtree(children[1] as usize);
                let balance = u32::abs_diff(l_leaves, r_leaves);
                let normalized_balance = balance as f64 / (l_leaves + r_leaves) as f64;
                balances_sum[nm as usize] += balance;
                norm_balances_sum[nm as usize] += normalized_balance;
                repetitions[nm as usize] += 1;
            }
        }

        let (balance_mean, normalized_balance_mean) = repetitions
            .iter()
            .zip(balances_sum.iter().zip(norm_balances_sum.iter()))
            .map(|(&r, (&b, &nb))| {
                if r > 0 {
                    let r = r as f64;
                    (b as f64 / r, nb / r)
                } else {
                    (0.0, 0.0)
                }
            })
            .collect();

        (balance_mean, normalized_balance_mean)
    }

    /// Calculate the pairwise distance distribution between `nodes`
    pub fn distance_dist<T: Count + Sync + Send>(&self, nodes: Range<usize>) -> Vec<T> {
        // Avoid calculating distances between the same node, so end at nodes.end - 1
        (nodes.start..(nodes.end - 1))
            .flat_map(|node_i| {
                let nm_i = self.total_mutations[node_i];
                ((node_i + 1)..nodes.end).map(move |node_j| {
                    let nm_j = self.total_mutations[node_j];
                    let lca = self.lca_query(node_i, node_j);
                    let nm_lca = self.total_mutations[lca];
                    nm_i + nm_j - 2 * nm_lca
                })
            })
            .into_bounded_iter(2 * self.max_n_mutations as usize)
            .freq()
    }

    /// Calculate the pairwise distance distribution between all leaves
    pub fn distance_dist_leaves<T: Count + Sync + Send>(&self) -> Vec<T> {
        self.distance_dist(1..(self.n_leaves + 1)) // 0 is reserved for the root
    }

    /// Calculate the pairwise distance distribution between all nodes in the tree
    pub fn distance_dist_with_ancestors<T: Count + Sync + Send>(&self) -> Vec<T> {
        self.distance_dist(0..self.total_mutations.len())
    }

    /// Calculate the pairwise distance distribution between ancestors only
    pub fn distance_dist_ancestors_only<T: Count + Sync + Send>(&self) -> Vec<T> {
        let offset = self.n_leaves + 1;
        // All non-root inner nodes
        let mut freq = self.distance_dist(offset..self.total_mutations.len());

        // Count distances root and other ancestors
        let root_total_mutation = self.total_mutations[0];
        for &n in &self.total_mutations[offset..] {
            freq[(n - root_total_mutation) as usize] += T::ONE;
        }

        freq
    }
}

#[derive(Clone, Debug, Default)]
struct Children([Option<NonZero<u32>>; 2]);

impl Children {
    /// Add a child to the children
    ///
    /// # Panics
    ///
    /// If trying to add child when the children is full (2 children).
    ///
    /// # Return
    ///
    /// Return true if the child is inserted successfully.
    /// Return false if the child is already in the children.
    fn add_child(&mut self, child: NonZero<u32>) -> bool {
        for c in self.0.iter_mut() {
            if c == &Some(child) {
                // Skip
                return false;
            } else if c.is_none() {
                *c = Some(child);
                return true;
            } // else continue
        }
        panic!("Children is full");
    }

    fn iter(&self) -> ChildrenIter<'_> {
        ChildrenIter(self.0.iter())
    }
}

struct ChildrenIter<'a>(std::slice::Iter<'a, Option<NonZero<u32>>>);

impl Iterator for ChildrenIter<'_> {
    type Item = NonZero<u32>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(Some(child)) = self.0.next() {
            Some(*child)
        } else {
            None
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use rand::{SeedableRng, rngs::SmallRng};

    use super::*;
    #[derive(Debug, Clone, Copy)]
    struct Const(u16);

    impl Distribution<u16> for Const {
        fn sample<R: rand::Rng + ?Sized>(&self, _: &mut R) -> u16 {
            self.0
        }
    }

    fn divide_at(cells: &mut Vec<LineageNode>, i: usize) {
        crate::divide_at(cells, i, &mut ());
    }

    fn rng() -> SmallRng {
        SmallRng::from_os_rng()
    }

    #[test]
    fn test_tree() {
        let mut cells = vec![LineageNode::default()];
        let cells_mut_ref = &mut cells;
        divide_at(cells_mut_ref, 0); // cell_r divide, [1, 2]
        divide_at(cells_mut_ref, 0); // cell_1 divide, [11, 2, 12]
        divide_at(cells_mut_ref, 1); // cell_2 divide, [11, 21, 12, 22]
        divide_at(cells_mut_ref, 0); // cell_11 divide, [111, 21, 12, 22, 112]
        divide_at(cells_mut_ref, 1); // cell_21 divide, [111, 211, 12, 22, 112, 212]
        let phylo: PhyloTree<8> = PhyloTree::builder(cells, Const(1)).build(&mut rng());
        assert_eq!(phylo.sfs(), vec![
            0, // mutation shared by no cells, always 0
            6, // each cell has one unique mutation
            2, // mutation of 11 and 21 are shared by [111, 112] and [211, 212] respectively
            2, // mutation of 1 and 2 are shared by [111, 112, 12] and [211, 212, 22]
        ]);
        assert_eq!(phylo.mbd(), vec![0, 0, 2, 4]);
        assert_eq!(phylo.umbd(), vec![1, 10]);
        assert_eq!(
            phylo.bbm(),
            (vec![0.0, 1.0, 0.0, 0.0], vec![0.0, 1.0 / 3.0, 0.0, 0.0])
        );
        let ddl: Vec<u32> = phylo.distance_dist_leaves();
        assert_eq!(ddl, vec![0, 0, 2, 4, 1, 4, 4]);
        let ddwa: Vec<u32> = phylo.distance_dist_with_ancestors();
        assert_eq!(ddwa, vec![0, 10, 13, 12, 8, 8, 4]);
        let dda: Vec<u32> = phylo.distance_dist_ancestors_only();
        assert_eq!(dda, vec![0, 4, 3, 2, 1]);
    }

    #[test]
    fn test_sample_cells() {
        let mut cells = vec![LineageNode::default()];
        let cells_mut_ref = &mut cells;
        divide_at(cells_mut_ref, 0); // cell_r divide, [1, 2]
        divide_at(cells_mut_ref, 0); // cell_1 divide, [11, 2, 12]
        divide_at(cells_mut_ref, 1); // cell_2 divide, [11, 21, 12, 22]
        divide_at(cells_mut_ref, 0); // cell_11 divide, [111, 21, 12, 22, 112]
        divide_at(cells_mut_ref, 1); // cell_21 divide, [111, 211, 12, 22, 112, 212]
        let phylo: PhyloTree<8> = PhyloTree::builder(cells, Const(1))
            .sample(&mut rng(), 3)
            .build(&mut rng());

        // We can't test the tree from sampled cells directly as we don't know which are sampled
        // Thus, here we only test some properties of the sampled tree
        assert_eq!(phylo.n_leaves, 3);
        assert_eq!(phylo.total_mutations.len(), 5);
        assert_eq!(phylo.euler_tour.len(), 9);
    }

    #[test]
    fn test_absorbing_node() {
        let mut cells = vec![LineageNode::default()];
        let cells_mut_ref = &mut cells;
        divide_at(cells_mut_ref, 0); // cell_r divide, [1, 2]
        divide_at(cells_mut_ref, 0); // cell_1 divide, [11, 2, 12]
        divide_at(cells_mut_ref, 1); // cell_2 divide, [11, 21, 12, 22]
        divide_at(cells_mut_ref, 0); // cell_11 divide, [111, 21, 12, 22, 112]
        divide_at(cells_mut_ref, 1); // cell_21 divide, [111, 211, 12, 22, 112, 212]
        cells_mut_ref.remove(0); // cell_111 died, [211, 12, 22, 112, 212]
        let phylo: PhyloTree<8> = PhyloTree::builder(cells, Const(1)).build(&mut rng());
        assert_eq!(phylo.sfs(), vec![
            0, // mutation shared by no cells, always 0
            6, // each cell has one unique mutation
            2, // mutation of 1, 21 are shared by [112, 12] and [211, 212] respectively
            1, // mutation of 2 are shared by [211, 212, 22]
        ]);
        assert_eq!(phylo.mbd(), vec![0, 0, 2, 3]);
        assert_eq!(phylo.umbd(), vec![1, 7, 1]);
        assert_eq!(
            phylo.bbm(),
            (vec![1.0, 0.5, 0.0, 0.0], vec![
                1.0 / 5.0,
                (1.0 / 3.0 + 0.0) / 2.0,
                0.0,
                0.0
            ])
        );
        let ddl: Vec<u32> = phylo.distance_dist_leaves();
        assert_eq!(ddl, vec![0, 0, 1, 3, 1, 3, 2]);
    }

    #[test]
    fn test_absorbing_root() {
        let mut cells = vec![LineageNode::default()];
        let cells_mut_ref = &mut cells;
        divide_at(cells_mut_ref, 0); // cell_r divide, [1, 2]
        divide_at(cells_mut_ref, 0); // cell_1 divide, [11, 2, 12]
        cells_mut_ref.remove(1); // cell_2 died, [11, 12]
        let phylo: PhyloTree<2> = PhyloTree::builder(cells, Const(1)).build(&mut rng());
        assert_eq!(phylo.sfs(), vec![
            0, // mutation shared by no cells, always 0
            2, // each cell has one unique mutation
            1, // mutation of 1 are shared by [11, 12]
        ]);
        assert_eq!(phylo.mbd(), vec![0, 0, 2]);
        assert_eq!(phylo.umbd(), vec![0, 3]);
        assert_eq!(phylo.bbm(), (vec![0.0, 0.0, 0.00], vec![0.0, 0.0, 0.0]));
        let ddl: Vec<u32> = phylo.distance_dist_leaves();
        assert_eq!(ddl, vec![0, 0, 1]);
        let ddwa: Vec<u32> = phylo.distance_dist_with_ancestors();
        assert_eq!(ddwa, vec![0, 2, 1]);
        let dda: Vec<u32> = phylo.distance_dist_ancestors_only();
        assert_eq!(dda, vec![]);
    }

    #[test]
    fn test_absorbing_root_with_observed_child() {
        let mut cells = vec![LineageNode::default()];
        let cells_mut_ref = &mut cells;
        divide_at(cells_mut_ref, 0); // cell_r divide, [1, 2]
        divide_at(cells_mut_ref, 0); // cell_1 divide, [11, 2, 12]
        cells_mut_ref.remove(1); // cell_2 died, [11, 12]
        divide_at(cells_mut_ref, 0); // cell_11 divide, [111, 12, 112]
        divide_at(cells_mut_ref, 1); // cell_12 divide, [111, 121, 112, 122]
        cells_mut_ref.remove(0); // cell_111 died, [121, 112, 122]
        let phylo: PhyloTree<2> = PhyloTree::builder(cells, Const(1)).build(&mut rng());
        assert_eq!(phylo.sfs(), vec![
            0, // mutation shared by no cells, always 0
            4, // each cell has one unique mutation, and 11 is observed in 112
            1, // mutation of 12 is shared by [121, 122]
            1, // mutation if 1 is shared by [112, 112, 122]
        ]);
        assert_eq!(phylo.mbd(), vec![0, 0, 0, 3]);
        assert_eq!(phylo.umbd(), vec![0, 4, 1]);
        assert_eq!(
            phylo.bbm(),
            (vec![0.0, 1.0, 0.0, 0.0], vec![0.0, 1.0 / 3.0, 0.0, 0.0])
        );
        let ddl: Vec<u32> = phylo.distance_dist_leaves();
        assert_eq!(ddl, vec![0, 0, 1, 0, 2]);
    }

    #[test]
    fn test_from_poisson() {
        let mut cells = vec![LineageNode::default()];
        let cells_mut_ref = &mut cells;
        divide_at(cells_mut_ref, 0); // cell_r divide, [1, 2]
        let dist = PoissonKnuth::new(10.0).unwrap();
        let mut rng = SmallRng::seed_from_u64(0);
        let m1 = dist.sample(&mut rng) as usize;
        let m2 = dist.sample(&mut rng) as usize;
        let mut rng = SmallRng::seed_from_u64(0);
        let phylo: PhyloTree<2> = PhyloTree::poisson_builder(cells, 10.0)
            .unwrap()
            .build(&mut rng);
        assert_eq!(phylo.sfs(), vec![0, (m1 + m2) as u32]);
        let mut expect_mbd = vec![0; m1.max(m2) + 1];
        expect_mbd[m1] += 1;
        expect_mbd[m2] += 1;
        assert_eq!(phylo.mbd(), expect_mbd);
        let mut expect_umbd = expect_mbd.clone();
        expect_umbd[0] += 1;
        assert_eq!(phylo.umbd(), expect_umbd);
        assert_eq!(
            phylo.bbm(),
            (vec![0.0; m1.max(m2) + 1], vec![0.0; m1.max(m2) + 1])
        );
        let mut expect_dd = vec![0; m1 + m2 + 1];
        expect_dd[m1 + m2] = 1;
        let ddl: Vec<u32> = phylo.distance_dist_leaves();
        assert_eq!(ddl, expect_dd);
    }

    #[cfg(feature = "bitcode")]
    #[test]
    fn test_bitcode_serialize() {
        let mut cells = vec![LineageNode::default()];
        let cells_mut_ref = &mut cells;
        divide_at(cells_mut_ref, 0); // cell_r divide, [1, 2]
        let phylo: PhyloTree<2> = PhyloTree::builder(cells, Const(1)).build(&mut rng());
        let serialized = bitcode::encode(&phylo);
        let deserialized: PhyloTree<2> = bitcode::decode(&serialized).unwrap();

        assert_eq!(deserialized.mbd(), phylo.mbd());
        assert_eq!(deserialized.umbd(), phylo.umbd());
        assert_eq!(deserialized.bbm(), phylo.bbm());
        let ddl: Vec<u32> = deserialized.distance_dist_leaves();
        assert_eq!(ddl, phylo.distance_dist_leaves());
    }
}
