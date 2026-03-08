//! Manual implementation of frequency counting algorithms.

use frequency::prelude::*;

pub fn bounded_freq<T: ToUsize + Copy, C: Count>(data: &[T], max_value: usize) -> Vec<C> {
    let mut freq = vec![C::ZERO; max_value + 1];
    for &val in data.iter() {
        freq[val.to_usize()] += C::ONE;
    }
    freq
}

/// # Safety
///
/// This function assumes that the values in `data` are within the range `[0, max_value]`.
/// If the values are outside this range, the behavior is undefined.
pub unsafe fn bounded_freq_unchecked<T: ToUsize + Copy, C: Count>(
    data: &[T],
    max_value: usize,
) -> Vec<C> {
    let mut freq = vec![C::ZERO; max_value + 1];
    for &val in data.iter() {
        unsafe {
            *freq.get_unchecked_mut(val.to_usize()) += C::ONE;
        }
    }
    freq
}

#[cfg(feature = "parallel")]
pub mod parallel {
    use std::num::NonZeroUsize;

    use rayon::prelude::*;

    use super::*;

    /// Parallel version of `bounded_freq`.
    ///
    /// Splits data into `n_threads` chunks and processes them on Rayon workers.
    pub fn par_bounded_freq<T, C>(data: &[T], max_value: usize, n_threads: NonZeroUsize) -> Vec<C>
    where
        T: ToUsize + Copy + Send + Sync,
        C: Count + Send,
    {
        let len = data.len();
        if len == 0 {
            return vec![C::ZERO; max_value + 1];
        }
        let n_threads = std::cmp::min(n_threads.get(), len);
        let chunk_size = len.div_ceil(n_threads);

        data.par_chunks(chunk_size)
            .map(|chunk| {
                let mut partial_freq = vec![C::ZERO; max_value + 1];
                for &val in chunk {
                    let index = val.to_usize();
                    if index <= max_value {
                        partial_freq[index] += C::ONE;
                    } else {
                        panic!("Value {index} exceeds max_value {max_value}");
                    }
                }
                partial_freq
            })
            .reduce(
                || vec![C::ZERO; max_value + 1],
                |mut final_freq, partial_freq| {
                    for (total, count) in final_freq.iter_mut().zip(partial_freq) {
                        *total += count;
                    }
                    final_freq
                },
            )
    }

    /// Parallel version of `unchecked_bounded_freq`.
    ///
    /// Splits data into `n_threads` chunks and processes them on Rayon workers.
    ///
    /// # Safety
    ///
    /// This function assumes that the values in `data` are within the range `[0, max_value]`.
    /// If the values are outside this range, the behavior is undefined.
    pub unsafe fn par_bounded_freq_unchecked<T, C>(
        data: &[T],
        max_value: usize,
        n_threads: NonZeroUsize,
    ) -> Vec<C>
    where
        T: ToUsize + Copy + Send + Sync,
        C: Count + Send,
    {
        let len = data.len();
        if len == 0 {
            return vec![C::ZERO; max_value + 1];
        }
        let n_threads = std::cmp::min(n_threads.get(), len);
        let chunk_size = len.div_ceil(n_threads);

        data.par_chunks(chunk_size)
            .map(|chunk| {
                let mut partial_freq = vec![C::ZERO; max_value + 1];
                for &val in chunk {
                    // SAFETY: Caller guarantees val.to_usize() <= max_value.
                    unsafe {
                        *partial_freq.get_unchecked_mut(val.to_usize()) += C::ONE;
                    }
                }
                partial_freq
            })
            .reduce(
                || vec![C::ZERO; max_value + 1],
                |mut final_freq, partial_freq| {
                    for (index, count) in partial_freq.into_iter().enumerate() {
                        // SAFETY: `index` is derived from a vector with `max_value + 1` entries.
                        unsafe {
                            *final_freq.get_unchecked_mut(index) += count;
                        }
                    }
                    final_freq
                },
            )
    }
}
