use rayon::prelude::*;

pub(crate) trait ParallelCount<R> {
    /// Count the occurrences of each element in a parallel iterator
    ///
    /// # Panic
    ///
    /// Panics if the iterator contains elements greater than `max_value`.
    fn par_count(self, max_value: usize) -> Vec<R>;
}

macro_rules! impl_parallel_count {
    ($elem:ident, $ret:ident) => {
        impl<I: ParallelIterator<Item = $elem>> ParallelCount<$ret> for I {
            fn par_count(self, max_value: usize) -> Vec<$ret> {
                let len = max_value + 1;
                let mut count = self
                    .fold(
                        || vec![0; len],
                        |mut acc, elem| {
                            acc[elem as usize] += 1;
                            acc
                        },
                    )
                    .reduce(
                        || vec![0; len],
                        |mut count1, count2| {
                            for (a1, a2) in count1.iter_mut().zip(count2) {
                                *a1 += a2;
                            }
                            count1
                        },
                    );

                remove_trailing(&mut count, &0);

                count
            }
        }
    };
}

impl_parallel_count!(u16, u16);
impl_parallel_count!(u16, u32);
impl_parallel_count!(u16, u64);

pub(crate) trait ParallelWeightedCount {
    type Weight;

    /// Count the occurrences of each element in a parallel iterator with given weight
    ///
    /// # Panic
    ///
    /// Panics if the iterator contains elements greater than `max_value`.
    fn par_weighted_count(self, max_value: usize) -> Vec<Self::Weight>;
}

macro_rules! impl_parallel_weighted_count {
    ($elem:ident, $weight:ident) => {
        impl<I: ParallelIterator<Item = ($elem, $weight)>> ParallelWeightedCount for I {
            type Weight = $weight;

            fn par_weighted_count(self, max_value: usize) -> Vec<$weight> {
                let len = max_value + 1;
                let mut count = self
                    .fold(
                        || vec![0; len],
                        |mut acc, (elem, weight)| {
                            acc[elem as usize] += weight;
                            acc
                        },
                    )
                    .reduce(
                        || vec![0; len],
                        |mut count1, count2| {
                            for (a1, a2) in count1.iter_mut().zip(count2) {
                                *a1 += a2;
                            }
                            count1
                        },
                    );

                remove_trailing(&mut count, &0);

                count
            }
        }
    };
}

impl_parallel_weighted_count!(u32, u32);

/// Remove trailing elements from a vector that match a given target
fn remove_trailing<T: PartialEq>(vec: &mut Vec<T>, target: &T) {
    let mut i = vec.len();
    for (j, val) in vec.iter().enumerate().rev() {
        if val != target {
            i = j + 1;
            break;
        }
    }
    vec.truncate(i);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_remove_trailing() {
        fn remove_trailing_zero(mut vec: Vec<usize>) -> Vec<usize> {
            remove_trailing(&mut vec, &0);
            vec
        }

        assert_eq!(remove_trailing_zero(vec![0, 0, 0, 1]), &[0, 0, 0, 1]);
        assert_eq!(remove_trailing_zero(vec![0, 0, 1, 0]), &[0, 0, 1]);
    }

    #[test]
    fn test_par_count() {
        let data: Vec<u16> = vec![1, 2, 3, 1, 2, 1];
        let result: Vec<u16> = data.into_par_iter().par_count(3);
        assert_eq!(result, [0, 3, 2, 1]);

        let data: Vec<u16> = vec![];
        let result: Vec<u32> = data.into_par_iter().par_count(0);
        assert_eq!(result, [0]);
    }
}
