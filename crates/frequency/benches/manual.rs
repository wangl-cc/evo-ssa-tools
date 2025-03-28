//! Manual implementation of frequency counting algorithms.

use frequency::prelude::*;

/// # Safety
///
/// This function assumes that the values in `data` are within the range `[0, max_value]`.
/// If the values are outside this range, the behavior is undefined.
pub unsafe fn unchecked_bounded_freq<T: ToUsize + Copy, C: Count>(
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

pub fn bounded_freq<T: ToUsize + Copy, C: Count>(data: &[T], max_value: usize) -> Vec<C> {
    let mut freq = vec![C::ZERO; max_value + 1];
    for &val in data.iter() {
        freq[val.to_usize()] += C::ONE;
    }
    freq
}
