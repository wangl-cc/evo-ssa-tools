//! This module provides implementations for calculating frequencies from iterators.

use core::hash::{BuildHasher, Hash};
use std::collections::HashMap;

use crate::{num_traits::*, *};

/// A wrapper around an iterator whose items implement the [`ToUsize`]
/// trait and have a known upper bound.
///
/// Bounded iterators are ideal for calculating the frequency of elements in a collection with a
/// dense distribution within a narrow range (e.g., counting dice rolls from 1-6).
///
/// # When to use
///
/// - Your values can be efficiently converted to usize indices.
/// - You know the possible range of values.
/// - Your values are densely populated within the range, or the range is small.
///
/// If your items don't meet these criteria, hashable iterators are a better choice.
/// See [`HashableIterator`].
///
/// # Tips
///
/// If your values are signed integers, or unsigned integers not starting at zero, consider shifting
/// them to start at zero to use this trait.
pub struct BoundedIterator<I: Iterator> {
    iter: I,
    upper_bound: usize,
}

/// A trait for converting an [`IntoIterator`] into a [`BoundedIterator`]
pub trait IntoBoundedIterator: IntoIterator {
    /// Converts the iterator into a [`BoundedIterator`]
    ///
    /// All items in the iterator must be within the range `0..=upper_bound`.
    fn into_bounded_iter(self, upper_bound: usize) -> BoundedIterator<Self::IntoIter>;
}

impl<I: IntoIterator> IntoBoundedIterator for I {
    fn into_bounded_iter(self, upper_bound: usize) -> BoundedIterator<I::IntoIter> {
        BoundedIterator {
            iter: self.into_iter(),
            upper_bound,
        }
    }
}

impl<E, C, I> Frequency<Vec<C>> for BoundedIterator<I>
where
    E: ToUsize,
    C: Count,
    I: Iterator<Item = E>,
{
    fn freq(self) -> Vec<C> {
        let len = self.upper_bound + 1;
        let mut freq = self.iter.fold(vec![C::ZERO; len], |mut freq, item| {
            freq[item.to_usize()] += C::ONE;
            freq
        });
        remove_trailing_zeros(&mut freq);
        freq
    }
}

impl<E, C, I> WeightedFrequency<Vec<C>> for BoundedIterator<I>
where
    E: ToUsize,
    C: Count,
    I: Iterator<Item = (E, C)>,
{
    /// Calculate the weighted frequency of each element in the bounded iterator.
    ///
    /// # Panics
    ///
    /// If there are elements outside the range [0, upper_bound], the function will panic.
    fn weighted_freq(self) -> Vec<C> {
        let len = self.upper_bound + 1;
        let mut freq = self
            .iter
            .fold(vec![C::ZERO; len], |mut freq, (item, weight)| {
                freq[item.to_usize()] += weight;
                freq
            });
        remove_trailing_zeros(&mut freq);
        freq
    }
}

/// An unsafe version of [`BoundedIterator`] that uses unchecked array access for performance.
///
/// # Safety
///
/// All items in the iterator must be within the range `0..=upper_bound`.
/// Using values outside this range will lead to undefined behavior.
pub struct UncheckedBoundedIterator<I: Iterator> {
    iter: I,
    upper_bound: usize,
}

/// A trait for converting an [`IntoIterator`] into an [`UncheckedBoundedIterator`]
pub trait IntoUncheckedBoundedIterator: IntoIterator {
    /// Converts the iterator into an [`UncheckedBoundedIterator`]
    ///
    /// # Safety
    ///
    /// All items in the iterator must be within the range `0..=upper_bound`.
    /// Using values outside this range will lead to undefined behavior.
    unsafe fn into_unchecked_bounded_iter(
        self,
        upper_bound: usize,
    ) -> UncheckedBoundedIterator<Self::IntoIter>;
}

impl<I: IntoIterator> IntoUncheckedBoundedIterator for I {
    unsafe fn into_unchecked_bounded_iter(
        self,
        upper_bound: usize,
    ) -> UncheckedBoundedIterator<I::IntoIter> {
        UncheckedBoundedIterator {
            iter: self.into_iter(),
            upper_bound,
        }
    }
}

impl<E, C, I> Frequency<Vec<C>> for UncheckedBoundedIterator<I>
where
    E: ToUsize,
    C: Count,
    I: Iterator<Item = E>,
{
    fn freq(self) -> Vec<C> {
        let len = self.upper_bound + 1;
        let mut freq = vec![C::ZERO; len];

        // SAFETY: We trust that all items are within bounds as documented
        unsafe {
            for item in self.iter {
                let idx = item.to_usize();
                *freq.get_unchecked_mut(idx) += C::ONE;
            }
        }

        remove_trailing_zeros(&mut freq);
        freq
    }
}

impl<E, C, I> WeightedFrequency<Vec<C>> for UncheckedBoundedIterator<I>
where
    E: ToUsize,
    C: Count,
    I: Iterator<Item = (E, C)>,
{
    fn weighted_freq(self) -> Vec<C> {
        let len = self.upper_bound + 1;
        let mut freq = vec![C::ZERO; len];

        // SAFETY: We trust that all items are within bounds as documented
        unsafe {
            for (item, weight) in self.iter {
                let idx = item.to_usize();
                *freq.get_unchecked_mut(idx) += weight;
            }
        }

        remove_trailing_zeros(&mut freq);
        freq
    }
}

/// A wrapper around an iterator whose items are hashable and comparable.
/// (i.e. implemented [`Hash`] and [`Eq`]).
///
/// This iterator can be used to calculate the frequency in more generic cases.
///
/// ## When to use
///
/// - Your values can't be converted to usize or the conversion is inefficient.
/// - Or, the range of possible values is unknown, or values are sparse distributed.
/// - Or, Your value are non-integer like types such as string, or custom structs.
///
/// For values with a small, known upper bound that can be converted to usize, consider using
/// bounded iterator instead for better performance. See [`BoundedIterator`] for more details.
///
/// ## Tips
///
/// The return frequencies is a general HashMap, so you can choose a high performance hasher to
/// improve performance by annotating the return hashmap type.
pub struct HashableIterator<I: Iterator>(I);

/// A trait for converting an [`IntoIterator`] into a [`HashableIterator`]
pub trait IntoHashableIterator: IntoIterator {
    fn into_hash_iter(self) -> HashableIterator<Self::IntoIter>;
}

impl<I: IntoIterator> IntoHashableIterator for I {
    fn into_hash_iter(self) -> HashableIterator<Self::IntoIter> {
        HashableIterator(self.into_iter())
    }
}

impl<E, C, I, S> Frequency<HashMap<E, C, S>> for HashableIterator<I>
where
    E: Eq + Hash,
    C: Count,
    I: Iterator<Item = E>,
    S: BuildHasher + Default,
{
    fn freq(self) -> HashMap<E, C, S> {
        self.0.fold(Default::default(), |mut freq, item| {
            *freq.entry(item).or_insert(C::ZERO) += C::ONE;
            freq
        })
    }
}

impl<E, C, I, S> WeightedFrequency<HashMap<E, C, S>> for HashableIterator<I>
where
    E: Eq + Hash,
    C: Count,
    I: Iterator<Item = (E, C)>,
    S: BuildHasher + Default,
{
    fn weighted_freq(self) -> HashMap<E, C, S> {
        self.0.fold(Default::default(), |mut freq, (item, weight)| {
            *freq.entry(item).or_insert(C::ZERO) += weight;
            freq
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounded_iterator_freq() {
        let data: Vec<usize> = vec![1, 2, 3, 2, 1, 4, 5, 3, 2];
        let freq: Vec<usize> = data.iter().map(|&x| x - 1).into_bounded_iter(4).freq();

        assert_eq!(freq, vec![2, 3, 2, 1, 1]);
    }

    #[test]
    fn test_bounded_iterator_weighted_freq() {
        let data: Vec<(usize, usize)> = vec![(1, 2), (2, 3), (3, 1), (1, 1)];
        let freq = data.into_bounded_iter(3).weighted_freq();

        assert_eq!(freq, vec![0, 3, 3, 1]);
    }

    #[test]
    fn test_unchecked_bounded_iterator_freq() {
        let data: Vec<usize> = vec![1, 2, 3, 2, 1, 4, 5, 3, 2];
        let freq: Vec<usize> = unsafe {
            data.iter()
                .map(|&x| x - 1)
                .into_unchecked_bounded_iter(4)
                .freq()
        };

        assert_eq!(freq, vec![2, 3, 2, 1, 1]);
    }

    #[test]
    fn test_unchecked_bounded_iterator_weighted_freq() {
        let data: Vec<(usize, usize)> = vec![(1, 2), (2, 3), (3, 1), (1, 1)];
        let freq: Vec<usize> = unsafe { data.into_unchecked_bounded_iter(3).weighted_freq() };

        assert_eq!(freq, vec![0, 3, 3, 1]);
    }

    #[test]
    fn test_hashable_iterator_freq() {
        let data = vec!["a", "b", "a", "c", "b", "a"];
        let freq: HashMap<&str, usize> = data.into_hash_iter().freq();

        assert_eq!(freq.get("a"), Some(&3));
        assert_eq!(freq.get("b"), Some(&2));
        assert_eq!(freq.get("c"), Some(&1));
        assert_eq!(freq.len(), 3);
    }

    #[test]
    fn test_hashable_iterator_weighted_freq() {
        let data = vec![("a", 2usize), ("b", 3), ("a", 1), ("c", 5)];
        let freq: HashMap<&str, usize> = data.into_hash_iter().weighted_freq();

        assert_eq!(freq.get("a"), Some(&3));
        assert_eq!(freq.get("b"), Some(&3));
        assert_eq!(freq.get("c"), Some(&5));
        assert_eq!(freq.len(), 3);
    }

    #[test]
    fn test_hashable_iterator_with_complex_types() {
        #[derive(Hash, Eq, PartialEq, Debug)]
        struct TestItem(u32, char);

        let data = vec![TestItem(1, 'a'), TestItem(2, 'b'), TestItem(1, 'a')];
        let freq: HashMap<TestItem, usize> = data.into_hash_iter().freq();

        assert_eq!(freq.get(&TestItem(1, 'a')), Some(&2));
        assert_eq!(freq.get(&TestItem(2, 'b')), Some(&1));
        assert_eq!(freq.len(), 2);
    }
}
