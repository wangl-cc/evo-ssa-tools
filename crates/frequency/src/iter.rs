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

/// A wrapper around an iterator of continuous values that bins them into
/// equal-width buckets over a fixed range.
///
/// This is useful for building histograms from floating-point data without
/// materializing all values first — memory usage is O(`n_bins`), not O(N).
///
/// # When to use
///
/// - Your values are continuous (e.g., f32/f64 distances, probabilities).
/// - You know the range of possible values.
/// - You want a fixed-resolution histogram.
///
/// Values outside `[min, max]` are clamped to the nearest boundary bin.
/// All arithmetic stays in the original precision `F`.
pub struct BinnedIterator<I: Iterator, F> {
    iter: I,
    n_bins: usize,
    min: F,
    inv_bin_width: F,
}

/// A trait for converting an [`IntoIterator`] into a [`BinnedIterator`]
pub trait IntoBinnedIterator: IntoIterator {
    /// Converts the iterator into a [`BinnedIterator`] that maps continuous
    /// values in `[min, max]` to `n_bins` equal-width bins.
    ///
    /// # Panics
    ///
    /// Panics if `n_bins == 0` or if `max <= min`.
    fn into_binned_iter<F: Float>(
        self,
        n_bins: usize,
        min: F,
        max: F,
    ) -> BinnedIterator<Self::IntoIter, F>;
}

impl<I: IntoIterator> IntoBinnedIterator for I {
    fn into_binned_iter<F: Float>(
        self,
        n_bins: usize,
        min: F,
        max: F,
    ) -> BinnedIterator<I::IntoIter, F> {
        assert!(n_bins > 0, "into_binned_iter requires at least one bin");
        assert!(max > min, "into_binned_iter requires max > min");

        BinnedIterator {
            iter: self.into_iter(),
            n_bins,
            min,
            inv_bin_width: F::from_usize(n_bins) / (max - min),
        }
    }
}

impl<F, C, I> Frequency<Vec<C>> for BinnedIterator<I, F>
where
    F: Float,
    C: Count,
    I: Iterator<Item = F>,
{
    fn freq(self) -> Vec<C> {
        let mut freq = vec![C::ZERO; self.n_bins];
        for value in self.iter {
            let bin = ((value - self.min) * self.inv_bin_width).as_usize();
            let bin = bin.min(self.n_bins - 1);
            freq[bin] += C::ONE;
        }
        freq
    }
}

impl<F, C, I> WeightedFrequency<Vec<C>> for BinnedIterator<I, F>
where
    F: Float,
    C: Count,
    I: Iterator<Item = (F, C)>,
{
    fn weighted_freq(self) -> Vec<C> {
        let mut freq = vec![C::ZERO; self.n_bins];
        for (value, weight) in self.iter {
            let bin = ((value - self.min) * self.inv_bin_width).as_usize();
            let bin = bin.min(self.n_bins - 1);
            freq[bin] += weight;
        }
        freq
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use std::collections::hash_map::RandomState;

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

    #[test]
    fn test_binned_iterator_freq() {
        // 4 bins over [0, 1): [0, 0.25), [0.25, 0.5), [0.5, 0.75), [0.75, 1.0]
        let data = vec![0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0];
        let freq: Vec<usize> = data.into_binned_iter(4, 0.0, 1.0).freq();
        // 0.0 → bin 0, 0.1 → bin 0, 0.3 → bin 1, 0.5 → bin 2, 0.7 → bin 2, 0.9 → bin 3, 1.0 →
        // clamped to bin 3
        assert_eq!(freq, vec![2, 1, 2, 2]);
    }

    #[test]
    fn test_binned_iterator_weighted_freq() {
        let data = vec![(0.1, 2.0f64), (0.6, 3.0), (0.9, 1.5)];
        let freq: Vec<f64> = data.into_binned_iter(2, 0.0, 1.0).weighted_freq();
        // 0.1 → bin 0 (weight 2.0), 0.6 → bin 1 (weight 3.0), 0.9 → bin 1 (weight 1.5)
        assert_eq!(freq, vec![2.0, 4.5]);
    }

    #[test]
    fn test_binned_iterator_clamps_boundaries() {
        // Values below min saturate to bin 0, values above max clamp to last bin
        let data = vec![-0.5, 0.5, 1.5];
        let freq: Vec<usize> = data.into_binned_iter(2, 0.0, 1.0).freq();
        // -0.5 → (−0.5) * 2 = −1.0 → as usize saturates to 0 → bin 0
        // 0.5  → 0.5 * 2 = 1.0 → as usize = 1 → bin 1
        // 1.5  → 1.5 * 2 = 3.0 → min(1) → bin 1
        assert_eq!(freq, vec![1, 2]);
    }

    #[test]
    fn test_binned_iterator_f32() {
        let data: Vec<f32> = vec![0.0, 0.3, 0.7, 1.0];
        let freq: Vec<usize> = data.into_binned_iter(2, 0.0, 1.0).freq();
        assert_eq!(freq, vec![2, 2]);
    }

    #[test]
    fn test_bounded_iterator_weighted_freq_with_float_weights() {
        let data = vec![(0usize, 0.5f64), (1, 1.5), (1, 2.0)];
        let freq: Vec<f64> = data.into_bounded_iter(1).weighted_freq();
        assert_eq!(freq, vec![0.5, 3.5]);
    }

    #[test]
    fn test_unchecked_bounded_iterator_weighted_freq_with_float_weights() {
        let data = vec![(0usize, 0.5f64), (1, 1.5), (1, 2.0)];
        let freq: Vec<f64> = unsafe { data.into_unchecked_bounded_iter(1).weighted_freq() };
        assert_eq!(freq, vec![0.5, 3.5]);
    }

    #[test]
    fn test_bounded_iterator_single_upper_bound() {
        let data = vec![0usize, 0, 0];
        let freq: Vec<usize> = data.into_bounded_iter(0).freq();
        assert_eq!(freq, vec![3]);
    }

    #[test]
    fn test_hashable_iterator_with_explicit_hasher_type() {
        let data = vec!["a", "b", "a"];
        let freq: HashMap<&str, usize, RandomState> = data.into_hash_iter().freq();
        assert_eq!(freq.get("a"), Some(&2));
        assert_eq!(freq.get("b"), Some(&1));
    }

    #[test]
    fn test_bounded_iterator_freq_with_float_counts() {
        let data = vec![0usize, 1, 1, 2];
        let freq: Vec<f64> = data.into_bounded_iter(2).freq();
        assert_eq!(freq, vec![1.0, 2.0, 1.0]);
    }

    #[test]
    fn test_empty_bounded_iterator_truncates_to_empty() {
        let freq: Vec<usize> = std::iter::empty::<usize>().into_bounded_iter(3).freq();
        assert_eq!(freq, Vec::<usize>::new());
    }

    #[test]
    fn test_empty_unchecked_bounded_iterator_truncates_to_empty() {
        let freq: Vec<usize> = unsafe {
            std::iter::empty::<usize>()
                .into_unchecked_bounded_iter(3)
                .freq()
        };
        assert_eq!(freq, Vec::<usize>::new());
    }

    #[test]
    fn test_empty_hashable_iterator_returns_empty_map() {
        let freq: HashMap<usize, usize> = std::iter::empty::<usize>().into_hash_iter().freq();
        assert!(freq.is_empty());
    }

    #[test]
    fn test_empty_binned_iterator_returns_zeroed_bins() {
        let freq: Vec<usize> = std::iter::empty::<f64>()
            .into_binned_iter(3, 0.0, 1.0)
            .freq();
        assert_eq!(freq, vec![0, 0, 0]);
    }

    #[test]
    fn test_binned_iterator_freq_with_float_counts() {
        let data = vec![0.1f64, 0.2, 0.9];
        let freq: Vec<f64> = data.into_binned_iter(2, 0.0, 1.0).freq();
        assert_eq!(freq, vec![2.0, 1.0]);
    }

    #[test]
    fn test_binned_iterator_weighted_freq_with_integer_weights() {
        let data = vec![(0.1f32, 2usize), (0.6f32, 3usize), (0.9f32, 1usize)];
        let freq: Vec<usize> = data.into_binned_iter(2, 0.0, 1.0).weighted_freq();
        assert_eq!(freq, vec![2, 4]);
    }

    #[test]
    fn test_empty_binned_iterator_weighted_returns_zeroed_bins() {
        let freq: Vec<f64> = std::iter::empty::<(f64, f64)>()
            .into_binned_iter(3, 0.0, 1.0)
            .weighted_freq();
        assert_eq!(freq, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    #[should_panic(expected = "into_binned_iter requires at least one bin")]
    fn test_binned_iterator_rejects_zero_bins() {
        let _ = vec![0.0].into_binned_iter(0, 0.0, 1.0);
    }

    #[test]
    #[should_panic(expected = "into_binned_iter requires max > min")]
    fn test_binned_iterator_rejects_equal_range_bounds() {
        let _ = vec![0.0].into_binned_iter(4, 1.0, 1.0);
    }

    #[test]
    #[should_panic(expected = "into_binned_iter requires max > min")]
    fn test_binned_iterator_rejects_reversed_range_bounds() {
        let _ = vec![0.0].into_binned_iter(4, 1.0, 0.0);
    }
}
