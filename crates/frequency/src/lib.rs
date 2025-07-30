#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
#![doc = include_str!("../README.md")]

pub mod prelude {
    //! Import of useful traits and types from the crate

    #[cfg(feature = "parallel")]
    pub use crate::par_iter::*;
    pub use crate::{Frequency, WeightedFrequency, iter::*, num_traits::*};
}

pub mod iter;
pub mod num_traits;
#[cfg(feature = "parallel")]
pub mod par_iter;

/// A trait for calculating frequencies of elements
///
/// This trait provides functionality to count the occurrences of each unique element in a
/// collection.
pub trait Frequency<R> {
    /// Calculate the frequency of each unique element in an iterator
    ///
    /// Returns a data structure containing unique elements from the iterator
    /// and the number of times each element appears.
    ///
    /// # Example
    ///
    /// ```
    /// use frequency::prelude::*;
    ///
    /// let items: Vec<usize> = vec![1, 2, 1, 3];
    /// let frequencies: Vec<usize> = items.into_bounded_iter(3).freq();
    /// assert_eq!(frequencies, &[0, 2, 1, 1]);
    /// ```
    fn freq(self) -> R;
}

/// A trait for calculating frequencies of elements with weights
///
/// This trait provides functionality to count the weighted occurrences of elements,
/// where each element can have an associated weight that contributes to its total frequency.
pub trait WeightedFrequency<R> {
    /// Count the frequency of each weighted item in an iterator
    ///
    /// Similar to `freq()`, but each element contributes its weight to the total
    /// rather than simply incrementing by 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use frequency::prelude::*;
    ///
    /// let weighted_data: Vec<(usize, f64)> = vec![(0, 0.5), (2, 1.5), (1, 2.0), (2, 1.0)];
    /// let frequencies = weighted_data
    ///     .iter()
    ///     .cloned()
    ///     .into_bounded_iter(2)
    ///     .weighted_freq();
    /// assert_eq!(frequencies, &[0.5, 2.0, 2.5]);
    /// ```
    fn weighted_freq(self) -> R;
}
