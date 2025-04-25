//! Manual implementation of frequency counting algorithms.

use std::{num::NonZeroUsize, thread};

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

/// Parallel version of `bounded_freq`.
///
/// Splits data into `n_threads` chunks and processes them concurrently.
pub fn par_bounded_freq<T, C>(data: &[T], max_value: usize, n_threads: NonZeroUsize) -> Vec<C>
where
    T: ToUsize + Copy + Send + Sync,
    C: Count + Send,
{
    let len = data.len();
    let n_threads = std::cmp::min(n_threads.get(), len);
    let chunk_size = len.div_ceil(n_threads); // Ceiling division

    let mut final_freq = vec![C::ZERO; max_value + 1];

    thread::scope(|s| {
        let mut handles = Vec::with_capacity(n_threads);

        for i in 0..n_threads {
            let start = i * chunk_size;
            let end = std::cmp::min(start + chunk_size, len);
            let chunk = &data[start..end];

            // Spawn a thread for each chunk
            let handle = s.spawn(move || {
                let mut partial_freq = vec![C::ZERO; max_value + 1];
                for &val in chunk {
                    let index = val.to_usize();
                    if index <= max_value {
                        // Bounds check within thread
                        partial_freq[index] += C::ONE;
                    } else {
                        // Panic if value exceeds max_value, consistent with sequential version
                        panic!("Value {} exceeds max_value {}", index, max_value);
                    }
                }
                partial_freq // Return partial result
            });
            handles.push(handle);
        }

        // Collect results from threads and aggregate
        for handle in handles {
            match handle.join() {
                Ok(partial_freq) => {
                    for (i, count) in partial_freq.into_iter().enumerate() {
                        if i <= max_value {
                            // Ensure index is valid before aggregating
                            final_freq[i] += count;
                        }
                    }
                }
                Err(e) => {
                    // Propagate panic from worker threads
                    std::panic::resume_unwind(e);
                }
            }
        }
    });

    final_freq
}

/// Parallel version of `unchecked_bounded_freq`.
/// Splits data into `n_threads` chunks and processes them concurrently.
///
/// # Safety
///
/// This function assumes that the values in `data` are within the range `[0, max_value]`.
/// If the values are outside this range, the behavior is undefined.
/// Requires `n_threads >= 1`.
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
    // Ensure n_threads is at least 1 and not more than data length if len > 0
    let n_threads = std::cmp::min(n_threads.get(), len);

    let chunk_size = len.div_ceil(n_threads); // Ceiling division

    thread::scope(|s| {
        let mut handles = Vec::with_capacity(n_threads);

        for i in 0..n_threads {
            let start = i * chunk_size;
            let end = std::cmp::min(start + chunk_size, len);
            let chunk = &data[start..end];

            // Spawn a thread for each chunk
            let handle = s.spawn(move || {
                let mut partial_freq = vec![C::ZERO; max_value + 1];
                for &val in chunk {
                    // SAFETY: Caller guarantees val.to_usize() <= max_value
                    unsafe {
                        *partial_freq.get_unchecked_mut(val.to_usize()) += C::ONE;
                    }
                }
                partial_freq // Return partial result
            });
            handles.push(handle);
        }

        // Collect results from threads and aggregate
        let mut final_freq = vec![C::ZERO; max_value + 1];
        for handle in handles {
            match handle.join() {
                Ok(partial_freq) => {
                    for (i, count) in partial_freq.into_iter().enumerate() {
                        // SAFETY: Indices derived from `partial_freq` are within bounds [0,
                        // max_value] because `partial_freq` was created
                        // with size `max_value + 1`. The aggregation `+=`
                        // assumes `C` handles potential overflows if not arbitrary precision.
                        unsafe {
                            *final_freq.get_unchecked_mut(i) += count;
                        }
                    }
                }
                Err(e) => {
                    // Propagate panic from worker threads
                    std::panic::resume_unwind(e);
                }
            }
        }

        final_freq
    })
}
