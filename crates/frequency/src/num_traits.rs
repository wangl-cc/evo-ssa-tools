//! This module contains some number traits used in frequency counting

/// A trait for safely and infallibly converting a type to a usize.
///
/// If two values will be converted to the same usize, we will treat them as the same element when
/// counting frequencies.
///
/// This trait is currently only implemented for unsigned integers whose bit width is less than or
/// equal to the system's pointer width. So this trait is not implemented for `u64` on 32 bits
/// systems, which will lead to wrong results, due to truncation.
///
/// For signed integers, you can transform them into unsigned integers by subtracting the minimum
/// value, and then casting it to unsigned integers with same bit width.
pub trait ToUsize {
    fn to_usize(self) -> usize;
}

macro_rules! impl_to_usize {
    ($($uint_type:ty),*) => {
        $(
            impl ToUsize for $uint_type {
                fn to_usize(self) -> usize {
                    self as usize
                }
            }
        )*
    };
}

impl_to_usize!(u8, u16, usize);

#[cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]
impl_to_usize!(u32);

#[cfg(target_pointer_width = "64")]
impl_to_usize!(u64);

/// A trait representing a type that can be used to count frequencies
///
/// This trait defines constants and operations required for frequency counting.
/// It's implemented for unsigned integers (for counting occurrences) and
/// floating-point numbers (for weighted counting).
pub trait Count: std::ops::AddAssign + Sized + Copy + PartialEq {
    const ZERO: Self;
    const ONE: Self;
}

macro_rules! impl_count_for_int {
    ($($int_type:ty),*) => {
        $(
            impl Count for $int_type {
                const ZERO: Self = 0;
                const ONE: Self = 1;
            }
        )*
    };
}

impl_count_for_int!(u8, u16, u32, u64, u128, usize);

macro_rules! impl_count_for_float {
    ($($float_type:ty),*) => {
        $(
            impl Count for $float_type {
                const ZERO: Self = 0.0;
                const ONE: Self = 1.0;
            }
        )*
    };
}

impl_count_for_float!(f32, f64);

/// Remove trailing zeros from a vector to reduce memory usage
pub(crate) fn remove_trailing_zeros<U: Count>(vec: &mut Vec<U>) {
    let mut i = vec.len();
    for (j, &val) in vec.iter().enumerate().rev() {
        if val != U::ZERO {
            i = j + 1;
            break;
        }
    }
    vec.truncate(i);
    vec.shrink_to_fit();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_usize() {
        assert_eq!(1u8.to_usize(), 1);
        assert_eq!(1u16.to_usize(), 1);
        #[cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]
        assert_eq!(1u32.to_usize(), 1);
        #[cfg(target_pointer_width = "64")]
        assert_eq!(1u64.to_usize(), 1);
        assert_eq!(1usize.to_usize(), 1);
    }

    #[test]
    fn test_remove_trailing() {
        fn rtz(mut vec: Vec<usize>) -> Vec<usize> {
            remove_trailing_zeros(&mut vec);
            vec
        }

        assert_eq!(rtz(vec![0, 0, 0, 1]), &[0, 0, 0, 1]);
        assert_eq!(rtz(vec![0, 0, 1, 0]), &[0, 0, 1]);
    }
}
