use crate::Result;

/// Mutable discrete sampler backed by a segment tree of cached weights.
#[derive(Debug, Clone, Default)]
pub(crate) struct SegmentTree {
    weights: Vec<f64>,
    tree: Vec<f64>,
    leaf_count: usize,
}

impl SegmentTree {
    pub(crate) fn len(&self) -> usize {
        self.weights.len()
    }

    pub(crate) fn total(&self) -> f64 {
        self.tree.get(1).copied().unwrap_or(0.0)
    }

    pub(crate) fn push(&mut self, weight: f64) {
        self.weights.push(weight);
        self.ensure_capacity(self.weights.len());
        self.set(self.weights.len() - 1, weight);
    }

    pub(crate) fn set(&mut self, index: usize, weight: f64) {
        assert!(index < self.weights.len(), "weight index is in bounds");
        self.weights[index] = weight;

        let mut tree_index = self.leaf_count + index;
        self.tree[tree_index] = weight;
        while tree_index > 1 {
            tree_index /= 2;
            self.tree[tree_index] = self.tree[tree_index * 2] + self.tree[tree_index * 2 + 1];
        }
    }

    /// Resize while evaluating weights only for newly appended entries.
    pub(crate) fn resize_with(
        &mut self,
        new_len: usize,
        mut new_weight: impl FnMut(usize) -> Result<f64>,
    ) -> Result<()> {
        let old_len = self.len();
        if new_len < old_len {
            self.weights.truncate(new_len);
            self.rebuild_tree();
            return Ok(());
        }

        for index in old_len..new_len {
            self.push(new_weight(index)?);
        }
        Ok(())
    }

    pub(crate) fn sample(&self, target: f64) -> Option<usize> {
        if target < 0.0 || target >= self.total() {
            return None;
        }

        let mut target = target;
        let mut tree_index = 1;
        while tree_index < self.leaf_count {
            let left = tree_index * 2;
            if target < self.tree[left] {
                tree_index = left;
            } else {
                target -= self.tree[left];
                tree_index = left + 1;
            }
        }

        let index = tree_index - self.leaf_count;
        (index < self.weights.len()).then_some(index)
    }

    fn ensure_capacity(&mut self, len: usize) {
        let required_leaf_count = len.next_power_of_two().max(1);
        if required_leaf_count <= self.leaf_count {
            return;
        }

        self.leaf_count = required_leaf_count;
        self.rebuild_tree();
    }

    fn rebuild_tree(&mut self) {
        if self.leaf_count == 0 {
            self.tree.clear();
            return;
        }

        self.tree.clear();
        self.tree.resize(self.leaf_count * 2, 0.0);
        for (index, weight) in self.weights.iter().copied().enumerate() {
            self.tree[self.leaf_count + index] = weight;
        }
        for index in (1..self.leaf_count).rev() {
            self.tree[index] = self.tree[index * 2] + self.tree[index * 2 + 1];
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn segment_tree_updates_total_and_samples() {
        let mut tree = SegmentTree::default();
        tree.push(1.0);
        tree.push(3.0);
        tree.push(2.0);

        assert_eq!(tree.total(), 6.0);
        assert_eq!(tree.sample(0.5), Some(0));
        assert_eq!(tree.sample(1.5), Some(1));
        assert_eq!(tree.sample(5.5), Some(2));

        tree.set(1, 0.0);
        assert_eq!(tree.total(), 3.0);
        assert_eq!(tree.sample(1.5), Some(2));
    }

    #[test]
    fn segment_tree_resizes_incrementally() -> Result<()> {
        let mut tree = SegmentTree::default();
        tree.resize_with(4, |index| Ok(index as f64 + 1.0))?;
        assert_eq!(tree.total(), 10.0);

        tree.resize_with(2, |_| unreachable!("shrinking does not evaluate weights"))?;
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.total(), 3.0);

        tree.resize_with(3, |_| Ok(5.0))?;
        assert_eq!(tree.total(), 8.0);
        Ok(())
    }
}
