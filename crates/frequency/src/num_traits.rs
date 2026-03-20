//! This module contains some number traits used in frequency counting

mod private {
    pub trait Sealed {}
}

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

/// A trait abstracting floating-point operations needed for binning.
pub trait Float:
    private::Sealed
    + Copy
    + PartialOrd
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
{
    fn from_usize(n: usize) -> Self;
    fn as_usize(self) -> usize;
}

/// A trait representing a type that can be used to count frequencies
///
/// This trait defines constants and operations required for frequency counting.
/// It's implemented for unsigned integers (for counting occurrences) and
/// floating-point numbers (for weighted counting).
pub trait Count: private::Sealed + std::ops::AddAssign + Sized + Copy + PartialEq {
    const ZERO: Self;
    const ONE: Self;

    /// Convert from a `usize` count.
    ///
    /// For integer types this is a wrapping `as` cast;
    /// for floating-point types it converts via `as`.
    fn from_count(n: usize) -> Self;
}

macro_rules! impl_count_for_int {
    ($($int_type:ty),*) => {
        $(
            impl private::Sealed for $int_type {}
            impl Count for $int_type {
                const ZERO: Self = 0;
                const ONE: Self = 1;
                fn from_count(n: usize) -> Self { n as Self }
            }
        )*
    };
}

impl_count_for_int!(u8, u16, u32, u64, u128, usize);

macro_rules! impl_count_and_float_for_float {
    ($($float_type:ty),*) => {
        $(
            impl private::Sealed for $float_type {}
            impl Count for $float_type {
                const ZERO: Self = 0.0;
                const ONE: Self = 1.0;
                fn from_count(n: usize) -> Self { n as Self }
            }

            impl Float for $float_type {
                #[inline]
                fn from_usize(n: usize) -> Self {
                    n as Self
                }

                #[inline]
                fn as_usize(self) -> usize {
                    self as usize
                }
            }
        )*
    };
}

impl_count_and_float_for_float!(f32, f64);

/// Remove trailing zeros from a vector to reduce memory usage
pub(crate) fn remove_trailing_zeros<U: Count>(vec: &mut Vec<U>) {
    let mut i = 0;
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
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    fn count_from_usize<C: Count>(n: usize) -> C {
        C::from_count(n)
    }

    fn float_from_usize<F: Float>(n: usize) -> F {
        F::from_usize(n)
    }

    fn float_to_usize<F: Float>(value: F) -> usize {
        value.as_usize()
    }

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
        assert_eq!(rtz(vec![0, 0]), &[]);
    }

    #[test]
    fn test_count_from_count_for_integers() {
        assert_eq!(count_from_usize::<u8>(3), 3);
        assert_eq!(count_from_usize::<u16>(3), 3);
        assert_eq!(count_from_usize::<u32>(3), 3);
        assert_eq!(count_from_usize::<u64>(3), 3);
        assert_eq!(count_from_usize::<u128>(3), 3);
        assert_eq!(count_from_usize::<usize>(3), 3);
    }

    #[test]
    fn test_count_from_count_for_floats() {
        assert_eq!(count_from_usize::<f32>(3), 3.0);
        assert_eq!(count_from_usize::<f64>(3), 3.0);
    }

    #[test]
    fn test_count_constants() {
        assert_eq!(<u8 as Count>::ZERO, 0);
        assert_eq!(<u8 as Count>::ONE, 1);
        assert_eq!(<f32 as Count>::ZERO, 0.0);
        assert_eq!(<f32 as Count>::ONE, 1.0);
    }

    #[test]
    fn test_float_conversions() {
        assert_eq!(float_from_usize::<f32>(5), 5.0);
        assert_eq!(float_from_usize::<f64>(5), 5.0);
        assert_eq!(float_to_usize(4.9f32), 4);
        assert_eq!(float_to_usize(4.9f64), 4);
    }

    #[test]
    fn test_float_conversions_at_zero() {
        assert_eq!(float_from_usize::<f32>(0), 0.0);
        assert_eq!(float_from_usize::<f64>(0), 0.0);
        assert_eq!(float_to_usize(0.0f32), 0);
        assert_eq!(float_to_usize(0.0f64), 0);
    }

    #[test]
    fn test_remove_trailing_for_float_counts() {
        fn rtz(mut vec: Vec<f64>) -> Vec<f64> {
            remove_trailing_zeros(&mut vec);
            vec
        }

        assert_eq!(rtz(vec![1.5, 0.0, 0.0]), &[1.5]);
        assert_eq!(rtz(vec![0.0, 2.5, 0.0]), &[0.0, 2.5]);
    }
}
