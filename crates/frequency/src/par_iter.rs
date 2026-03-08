//! This module provides implementations for calculating frequencies from parallel iterators.

use core::hash::{BuildHasher, Hash};
use std::collections::HashMap;

use rayon::{
    iter::{IndexedParallelIterator, MinLen},
    prelude::*,
};

use crate::{num_traits::*, *};

fn thread_sized_min_len(len: usize) -> usize {
    let threads = rayon::current_num_threads().max(1);
    len.div_ceil(threads).max(1)
}

/// An parallel iterator adapter for items with a known upper bound
///
/// See [`crate::iter::BoundedIterator`] for more details.
pub struct BoundedParallelIterator<I: ParallelIterator> {
    iter: I,
    upper_bound: usize,
}

/// A trait for converting an [`IntoParallelIterator`] into a [`BoundedParallelIterator`]
pub trait IntoBoundedParallelIterator: IntoParallelIterator {
    fn into_bounded_par_iter(self, upper_bound: usize) -> BoundedParallelIterator<Self::Iter>;
}

/// A trait for converting an indexed [`IntoParallelIterator`] into a [`BoundedParallelIterator`]
/// with a coarser default split strategy.
pub trait IntoBoundedIndexedParallelIterator: IntoParallelIterator
where
    Self::Iter: IndexedParallelIterator,
{
    fn into_bounded_indexed_par_iter(
        self,
        upper_bound: usize,
    ) -> BoundedParallelIterator<MinLen<Self::Iter>>;
}

/// A trait for converting a [`ParallelIterator`] into a [`BoundedParallelIterator`]
impl<I: IntoParallelIterator> IntoBoundedParallelIterator for I {
    /// Converts the parallel iterator into a [`BoundedParallelIterator`]
    ///
    /// All items in the iterator must be within the range `0..=upper_bound`.
    fn into_bounded_par_iter(self, upper_bound: usize) -> BoundedParallelIterator<Self::Iter> {
        BoundedParallelIterator {
            iter: self.into_par_iter(),
            upper_bound,
        }
    }
}

impl<I> IntoBoundedIndexedParallelIterator for I
where
    I: IntoParallelIterator,
    I::Iter: IndexedParallelIterator,
{
    /// Converts the indexed parallel iterator into a [`BoundedParallelIterator`] with roughly one
    /// chunk per Rayon worker.
    ///
    /// This reduces per-job frequency table allocation and merge overhead for dense bounded
    /// counting workloads.
    fn into_bounded_indexed_par_iter(
        self,
        upper_bound: usize,
    ) -> BoundedParallelIterator<MinLen<Self::Iter>> {
        let iter = self.into_par_iter();
        let min_len = thread_sized_min_len(iter.len());

        BoundedParallelIterator {
            iter: iter.with_min_len(min_len),
            upper_bound,
        }
    }
}

impl<E, U, I> Frequency<Vec<U>> for BoundedParallelIterator<I>
where
    E: ToUsize,
    U: Count + Send,
    I: ParallelIterator<Item = E>,
{
    fn freq(self) -> Vec<U> {
        let len = self.upper_bound + 1;
        let mut freq = self
            .iter
            .fold(
                || vec![U::ZERO; len],
                |mut acc, elem| {
                    acc[elem.to_usize()] += U::ONE;
                    acc
                },
            )
            .reduce(
                || vec![U::ZERO; len],
                |mut count1, count2| {
                    for (a1, a2) in count1.iter_mut().zip(count2) {
                        *a1 += a2;
                    }
                    count1
                },
            );

        remove_trailing_zeros(&mut freq);

        freq
    }
}

impl<E, U, I> WeightedFrequency<Vec<U>> for BoundedParallelIterator<I>
where
    E: ToUsize,
    U: Count + Send,
    I: ParallelIterator<Item = (E, U)>,
{
    fn weighted_freq(self) -> Vec<U> {
        let len = self.upper_bound + 1;
        let mut freq = self
            .iter
            .fold(
                || vec![U::ZERO; len],
                |mut acc, (elem, weight)| {
                    acc[elem.to_usize()] += weight;
                    acc
                },
            )
            .reduce(
                || vec![U::ZERO; len],
                |mut count1, count2| {
                    for (a1, a2) in count1.iter_mut().zip(count2) {
                        *a1 += a2;
                    }
                    count1
                },
            );

        remove_trailing_zeros(&mut freq);

        freq
    }
}

/// An unsafe version of [`BoundedParallelIterator`] that uses unchecked array access for
/// performance.
///
/// # Safety
///
/// All items in the iterator must be within the range `0..=upper_bound`.
/// Using values outside this range will lead to undefined behavior.
pub struct UncheckedBoundedParallelIterator<I: ParallelIterator> {
    iter: I,
    upper_bound: usize,
}

/// A trait for converting an [`IntoParallelIterator`] into an [`UncheckedBoundedParallelIterator`]
pub trait IntoUncheckedBoundedParallelIterator: IntoParallelIterator {
    /// Converts the iterator into an [`UncheckedBoundedParallelIterator`].
    ///
    /// # Safety
    ///
    /// All items in the iterator must be within the range `0..=upper_bound`.
    /// Using values outside this range will lead to undefined behavior.
    unsafe fn into_unchecked_bounded_par_iter(
        self,
        upper_bound: usize,
    ) -> UncheckedBoundedParallelIterator<Self::Iter>;
}

/// A trait for converting an indexed [`IntoParallelIterator`] into an
/// [`UncheckedBoundedParallelIterator`] with a coarser default split strategy.
pub trait IntoUncheckedBoundedIndexedParallelIterator: IntoParallelIterator
where
    Self::Iter: IndexedParallelIterator,
{
    /// Converts the indexed iterator into an [`UncheckedBoundedParallelIterator`] with roughly one
    /// chunk per Rayon worker.
    ///
    /// # Safety
    ///
    /// All items in the iterator must be within the range `0..=upper_bound`.
    /// Using values outside this range will lead to undefined behavior.
    unsafe fn into_unchecked_bounded_indexed_par_iter(
        self,
        upper_bound: usize,
    ) -> UncheckedBoundedParallelIterator<MinLen<Self::Iter>>;
}

impl<I: IntoParallelIterator> IntoUncheckedBoundedParallelIterator for I {
    /// # Safety
    ///
    /// All items in the iterator must be within the range `0..=upper_bound`.
    /// Using values outside this range will lead to undefined behavior.
    unsafe fn into_unchecked_bounded_par_iter(
        self,
        upper_bound: usize,
    ) -> UncheckedBoundedParallelIterator<Self::Iter> {
        UncheckedBoundedParallelIterator {
            iter: self.into_par_iter(),
            upper_bound,
        }
    }
}

impl<I> IntoUncheckedBoundedIndexedParallelIterator for I
where
    I: IntoParallelIterator,
    I::Iter: IndexedParallelIterator,
{
    unsafe fn into_unchecked_bounded_indexed_par_iter(
        self,
        upper_bound: usize,
    ) -> UncheckedBoundedParallelIterator<MinLen<Self::Iter>> {
        let iter = self.into_par_iter();
        let min_len = thread_sized_min_len(iter.len());

        UncheckedBoundedParallelIterator {
            iter: iter.with_min_len(min_len),
            upper_bound,
        }
    }
}

impl<E, U, I> Frequency<Vec<U>> for UncheckedBoundedParallelIterator<I>
where
    E: ToUsize,
    U: Count + Send,
    I: ParallelIterator<Item = E>,
{
    fn freq(self) -> Vec<U> {
        let len = self.upper_bound + 1;
        let mut freq = self
            .iter
            .fold(
                || vec![U::ZERO; len],
                |mut acc, elem| {
                    // SAFETY: `elem` is within range because of the contract of
                    // `UncheckedBoundedParallelIterator`
                    unsafe {
                        *acc.get_unchecked_mut(elem.to_usize()) += U::ONE;
                    }
                    acc
                },
            )
            .reduce(
                || vec![U::ZERO; len],
                |mut count1, count2| {
                    for (a1, a2) in count1.iter_mut().zip(count2) {
                        *a1 += a2;
                    }
                    count1
                },
            );

        remove_trailing_zeros(&mut freq);

        freq
    }
}

impl<E, U, I> WeightedFrequency<Vec<U>> for UncheckedBoundedParallelIterator<I>
where
    E: ToUsize,
    U: Count + Send,
    I: ParallelIterator<Item = (E, U)>,
{
    fn weighted_freq(self) -> Vec<U> {
        let len = self.upper_bound + 1;
        let mut freq = self
            .iter
            .fold(
                || vec![U::ZERO; len],
                |mut acc, (elem, weight)| {
                    // SAFETY: `elem` is within range because of the contract of
                    // `UncheckedBoundedParallelIterator`
                    unsafe {
                        *acc.get_unchecked_mut(elem.to_usize()) += weight;
                    }
                    acc
                },
            )
            .reduce(
                || vec![U::ZERO; len],
                |mut count1, count2| {
                    for (a1, a2) in count1.iter_mut().zip(count2) {
                        *a1 += a2;
                    }
                    count1
                },
            );

        remove_trailing_zeros(&mut freq);

        freq
    }
}

/// A parallel iterator adapter for arbitrary hashable types
///
/// See [`crate::iter::HashableIterator`] for more details.
pub struct HashableParallelIterator<I: ParallelIterator>(I);

/// A trait for converting an [`IntoParallelIterator`] into a [`HashableParallelIterator`]
pub trait IntoHashableParallelIterator: IntoParallelIterator {
    fn into_hash_per_iter(self) -> HashableParallelIterator<Self::Iter>;
}

/// A trait for converting an indexed [`IntoParallelIterator`] into a
/// [`HashableParallelIterator`] with a coarser default split strategy.
pub trait IntoHashableIndexedParallelIterator: IntoParallelIterator
where
    Self::Iter: IndexedParallelIterator,
{
    fn into_hash_indexed_par_iter(self) -> HashableParallelIterator<MinLen<Self::Iter>>;
}

impl<I: IntoParallelIterator> IntoHashableParallelIterator for I {
    fn into_hash_per_iter(self) -> HashableParallelIterator<Self::Iter> {
        HashableParallelIterator(self.into_par_iter())
    }
}

impl<I> IntoHashableIndexedParallelIterator for I
where
    I: IntoParallelIterator,
    I::Iter: IndexedParallelIterator,
{
    fn into_hash_indexed_par_iter(self) -> HashableParallelIterator<MinLen<Self::Iter>> {
        let iter = self.into_par_iter();
        let min_len = thread_sized_min_len(iter.len());

        HashableParallelIterator(iter.with_min_len(min_len))
    }
}

impl<E, U, I, S> Frequency<HashMap<E, U, S>> for HashableParallelIterator<I>
where
    E: Eq + Hash + Send,
    U: Count + Send,
    I: ParallelIterator<Item = E>,
    S: BuildHasher + Default + Send,
{
    fn freq(self) -> HashMap<E, U, S> {
        self.0
            .fold(HashMap::<E, U, S>::default, |mut acc, elem| {
                *acc.entry(elem).or_insert(U::ZERO) += U::ONE;
                acc
            })
            .reduce(Default::default, |map1, map2| {
                // Merge the smaller map into the larger one for efficiency
                let (smaller, mut larger) = if map1.len() < map2.len() {
                    (map1, map2)
                } else {
                    (map2, map1)
                };

                for (key, value) in smaller {
                    *larger.entry(key).or_insert(U::ZERO) += value;
                }
                larger
            })
    }
}

impl<E, U, I, S> WeightedFrequency<HashMap<E, U, S>> for HashableParallelIterator<I>
where
    E: Eq + Hash + Send,
    U: Count + Send,
    I: ParallelIterator<Item = (E, U)>,
    S: BuildHasher + Default + Send,
{
    fn weighted_freq(self) -> HashMap<E, U, S> {
        self.0
            .fold(HashMap::<E, U, S>::default, |mut acc, (elem, weight)| {
                *acc.entry(elem).or_insert(U::ZERO) += weight;
                acc
            })
            .reduce(Default::default, |map1, map2| {
                // Merge the smaller map into the larger one for efficiency
                let (smaller, mut larger) = if map1.len() < map2.len() {
                    (map1, map2)
                } else {
                    (map2, map1)
                };

                for (key, value) in smaller {
                    *larger.entry(key).or_insert(U::ZERO) += value;
                }
                larger
            })
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn test_bounded_parallel_iterator_freq() {
        let data: Vec<usize> = vec![1, 2, 3, 2, 1, 4, 5, 3, 2];
        let freq: Vec<usize> = data
            .par_iter()
            .map(|&x| x - 1)
            .into_bounded_par_iter(4)
            .freq();

        assert_eq!(freq, vec![2, 3, 2, 1, 1]);
    }

    #[test]
    fn test_bounded_parallel_iterator_weighted_freq() {
        let data = vec![(1usize, 2usize), (2, 3), (3, 1), (1, 1)];
        let freq: Vec<usize> = data.into_bounded_par_iter(3).weighted_freq();

        assert_eq!(freq, vec![0, 3, 3, 1]);
    }

    #[test]
    fn test_bounded_indexed_parallel_iterator_freq() {
        let data: Vec<usize> = vec![1, 2, 3, 2, 1, 4, 5, 3, 2];
        let freq: Vec<usize> = data
            .par_iter()
            .map(|&x| x - 1)
            .into_bounded_indexed_par_iter(4)
            .freq();

        assert_eq!(freq, vec![2, 3, 2, 1, 1]);
    }

    #[test]
    fn test_unchecked_bounded_parallel_iterator_freq() {
        let data: Vec<usize> = vec![1, 2, 3, 2, 1, 4, 5, 3, 2];
        let freq: Vec<usize> = unsafe {
            data.par_iter()
                .map(|&x| x - 1)
                .into_unchecked_bounded_par_iter(4)
                .freq()
        };

        assert_eq!(freq, vec![2, 3, 2, 1, 1]);
    }

    #[test]
    fn test_unchecked_bounded_indexed_parallel_iterator_freq() {
        let data: Vec<usize> = vec![1, 2, 3, 2, 1, 4, 5, 3, 2];
        let freq: Vec<usize> = unsafe {
            data.par_iter()
                .map(|&x| x - 1)
                .into_unchecked_bounded_indexed_par_iter(4)
                .freq()
        };

        assert_eq!(freq, vec![2, 3, 2, 1, 1]);
    }

    #[test]
    fn test_unchecked_bounded_parallel_iterator_weighted_freq() {
        let data = vec![(1usize, 2usize), (2, 3), (3, 1), (1, 1)];
        let freq: Vec<usize> = unsafe { data.into_unchecked_bounded_par_iter(3).weighted_freq() };

        assert_eq!(freq, vec![0, 3, 3, 1]);
    }

    #[test]
    fn test_hashable_parallel_iterator_freq() {
        let data = vec!["a", "b", "a", "c", "b", "a"];
        let freq: HashMap<&str, usize> = data.into_hash_per_iter().freq();

        assert_eq!(freq.get("a"), Some(&3));
        assert_eq!(freq.get("b"), Some(&2));
        assert_eq!(freq.get("c"), Some(&1));
        assert_eq!(freq.len(), 3);
    }

    #[test]
    fn test_hashable_parallel_iterator_weighted_freq() {
        let data = vec![("a", 2usize), ("b", 3), ("a", 1), ("c", 5)];
        let freq: HashMap<&str, usize> = data.into_hash_per_iter().weighted_freq();

        assert_eq!(freq.get("a"), Some(&3));
        assert_eq!(freq.get("b"), Some(&3));
        assert_eq!(freq.get("c"), Some(&5));
        assert_eq!(freq.len(), 3);
    }

    #[test]
    fn test_hashable_indexed_parallel_iterator_freq() {
        let data = vec!["a", "b", "a", "c", "b", "a"];
        let freq: HashMap<&str, usize> =
            data.par_iter().copied().into_hash_indexed_par_iter().freq();

        assert_eq!(freq.get("a"), Some(&3));
        assert_eq!(freq.get("b"), Some(&2));
        assert_eq!(freq.get("c"), Some(&1));
        assert_eq!(freq.len(), 3);
    }

    #[test]
    fn test_hashable_indexed_parallel_iterator_weighted_freq() {
        let data = vec![("a", 2usize), ("b", 3), ("a", 1), ("c", 5)];
        let freq: HashMap<&str, usize> = data
            .par_iter()
            .copied()
            .into_hash_indexed_par_iter()
            .weighted_freq();

        assert_eq!(freq.get("a"), Some(&3));
        assert_eq!(freq.get("b"), Some(&3));
        assert_eq!(freq.get("c"), Some(&5));
        assert_eq!(freq.len(), 3);
    }

    #[test]
    fn test_hashable_parallel_iterator_with_complex_types() {
        #[derive(Hash, PartialEq, Eq, Debug)]
        struct TestItem(u32, char);

        let data = vec![TestItem(1, 'a'), TestItem(2, 'b'), TestItem(1, 'a')];
        let freq: HashMap<TestItem, usize> = data.into_hash_per_iter().freq();

        assert_eq!(freq.get(&TestItem(1, 'a')), Some(&2));
        assert_eq!(freq.get(&TestItem(2, 'b')), Some(&1));
        assert_eq!(freq.len(), 2);
    }
}
