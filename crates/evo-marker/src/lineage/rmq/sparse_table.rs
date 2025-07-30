//! Implementation of Sparse Table algorithm for RMQ

use super::MinAndPos;

/// Sparse table for RMQ
#[derive(Debug, Clone)]
#[cfg_attr(feature = "bitcode", derive(bitcode::Decode, bitcode::Encode))]
pub struct RmqSpareTable<T: Clone + Copy + Ord> {
    len: usize,
    tables: Vec<Vec<MinAndPos<T, u32>>>,
}

impl<T: Clone + Copy + Ord + std::fmt::Debug> RmqSpareTable<T> {
    /// Create a new sparse table with given values
    ///
    /// # Panics
    ///
    /// If the input vector is less than or equal to 1.
    pub fn new(values: &[T]) -> Self {
        let len = values.len();
        assert!(len > 1, "Input vector must have more than one element");

        let n_tables = len.ilog2() as usize;

        let mut tables = Vec::with_capacity(n_tables);

        // Calculate for pairwise for adjacent elements
        let len_k = len - 1;
        let mut mins_k = Vec::with_capacity(len_k);
        for i in 0..len_k {
            let a = values[i];
            let b = values[i + 1];
            if a <= b {
                mins_k.push(MinAndPos::new(a, i as u32));
            } else {
                mins_k.push(MinAndPos::new(b, i as u32 + 1));
            }
        }
        tables.push(mins_k);

        // Calculate all higher powers of 2
        #[allow(
            clippy::needless_range_loop,
            reason = "The initial length is equal to n_tables"
        )]
        for k in 1..n_tables {
            let block_size_k = 1 << k;
            let mins_k = &tables[k - 1];
            let block_size_kp1 = 1 << (k + 1);
            let len_kp1 = len - block_size_kp1 + 1;
            let mut mins_kp1 = Vec::with_capacity(len_kp1);

            // Give compiler a hint that i and i + ref_block_size are within bounds
            assert!(len_kp1 + block_size_k <= mins_k.len());
            for i in 0..len_kp1 {
                let min_left_half = mins_k[i];
                let min_right_half = mins_k[i + block_size_k];
                let min = std::cmp::min(min_left_half, min_right_half);
                mins_kp1.push(min);
            }
            tables.push(mins_kp1);
        }

        Self { len, tables }
    }

    pub fn min_in(&self, start: u32, end: u32) -> MinAndPos<T, u32> {
        // This function is very hot, so we want to minimize the number of operations and branches,
        // so only check the start and end values on debug builds
        debug_assert!(start < end, "Start `{start}` must be less than end `{end}`");
        debug_assert!(
            end < self.len as u32,
            "End `{end}` must be less than length {}",
            self.len
        );

        let len = end - start + 1;
        let k = len.ilog2() as usize - 1;

        // If length is a power of 2, we can use the precomputed block directly
        if len.is_power_of_two() {
            return self.tables[k][start as usize];
        }

        // Otherwise, we need to find the minimum of two overlapping blocks
        let block_size = 1 << (k + 1);
        let left_min = self.tables[k][start as usize];
        let right_min = self.tables[k][end as usize - block_size + 1];

        std::cmp::min(left_min, right_min)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    #[should_panic]
    fn test_single_element() {
        let values = vec![42];
        let _rmq = RmqSpareTable::new(&values);
    }

    #[test]
    fn test_small_array() {
        let values = vec![3, 1, 4, 1, 5, 9];
        let rmq = RmqSpareTable::new(&values);

        // Test minimum in the entire range
        let min = rmq.min_in(0, 5);
        assert_eq!(min.val(), 1);
        assert!(min.pos() == 1 || min.pos() == 3);

        // Test minimum in a subrange
        let min = rmq.min_in(2, 5);
        assert_eq!(min.val(), 1);
        assert_eq!(min.pos(), 3);
    }

    #[test]
    fn test_descending_array() {
        let values = vec![9, 8, 7, 6, 5, 4, 3, 2, 1];
        let rmq = RmqSpareTable::new(&values);

        // Test minimum in different ranges
        assert_eq!(rmq.min_in(0, 8).val(), 1);
        assert_eq!(rmq.min_in(0, 8).pos(), 8);

        assert_eq!(rmq.min_in(0, 4).val(), 5);
        assert_eq!(rmq.min_in(0, 4).pos(), 4);

        assert_eq!(rmq.min_in(2, 6).val(), 3);
        assert_eq!(rmq.min_in(2, 6).pos(), 6);
    }

    #[test]
    fn test_power_of_two_length() {
        let values = vec![7, 2, 9, 1, 8, 3, 6, 4];
        let rmq = RmqSpareTable::new(&values);

        // Test ranges with lengths that are powers of 2
        assert_eq!(rmq.min_in(0, 7).val(), 1);
        assert_eq!(rmq.min_in(0, 7).pos(), 3);

        assert_eq!(rmq.min_in(0, 3).val(), 1);
        assert_eq!(rmq.min_in(0, 3).pos(), 3);

        assert_eq!(rmq.min_in(4, 7).val(), 3);
        assert_eq!(rmq.min_in(4, 7).pos(), 5);

        assert_eq!(rmq.min_in(2, 3).val(), 1);
        assert_eq!(rmq.min_in(2, 3).pos(), 3);
    }
}
