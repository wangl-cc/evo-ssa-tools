/// Canonical key format version embedded in every persistent namespace.
///
/// Bumping this constant isolates all new cache entries from namespaces written under an older
/// key encoding.
pub(crate) const CANONICAL_KEY_FORMAT: &str = "keyfmt-v2";

/// Encode a key into canonical bytes.
///
/// Implementations should be deterministic and consistent across different builds and runs, and
/// must always write exactly `Self::SIZE` bytes.
///
/// # Portability
///
/// `ssa-workflow` targets 64-bit platforms only. This avoids platform-dependent key encodings
/// such as `usize` width and makes cache keys stable across builds.
///
/// # Order-preserving guarantee
///
/// Built-in numeric encodings are lexicographically order-preserving: if `a < b` in the natural
/// numeric order, then `encode(a) < encode(b)` in unsigned byte-lexicographic order. This allows
/// storage backends to preserve numeric sweep locality when walking keys.
///
/// - Unsigned integers: big-endian bytes (naturally order-preserving).
/// - Signed integers: big-endian two's-complement bytes with the most-significant sign bit flipped.
/// - Floating-point values: NaN payloads are normalized to one canonical NaN and `-0.0` is treated
///   as `+0.0`. The resulting canonical order is: `-inf < finite negatives < 0 < finite positives <
///   +inf < canonical NaN`.
/// - Tuples and arrays: concatenation of element encodings; lexicographic ordering is inherited
///   whenever every component encoding is order-preserving.
///
/// User-defined `CanonicalEncode` implementations are not required to be order-preserving, but
/// should prefer it when the type has a natural order.
pub trait CanonicalEncode {
    const SIZE: usize;

    /// Encode self into the provided buffer.
    ///
    /// ## Safety
    ///
    /// The buffer must have length at least `Self::SIZE`.
    /// Implementations must only access `buffer[..Self::SIZE]`.
    unsafe fn encode_into(&self, buffer: &mut [u8]);

    /// Encode self into the provided buffer and return the encoded bytes.
    ///
    /// This is a convenience wrapper around `encode_into` that returns the encoded bytes.
    ///
    /// ## Safety
    ///
    /// The buffer must have length at least `Self::SIZE`.
    /// Implementations must only access `buffer[..Self::SIZE]`.
    unsafe fn encode_with_buffer<'b>(&self, buffer: &'b mut [u8]) -> &'b [u8] {
        unsafe { self.encode_into(buffer) };
        unsafe { buffer.get_unchecked(..Self::SIZE) }
    }
}

impl CanonicalEncode for () {
    const SIZE: usize = 0;

    #[inline]
    unsafe fn encode_into(&self, _buffer: &mut [u8]) {}
}

impl CanonicalEncode for bool {
    const SIZE: usize = 1;

    #[inline]
    unsafe fn encode_into(&self, buffer: &mut [u8]) {
        unsafe { *buffer.get_unchecked_mut(0) = *self as u8 };
    }
}

/// Sequential writer for building a [`CanonicalEncode`] implementation field by field.
///
/// This helper narrows the provided buffer to exactly `T::SIZE` bytes and then advances an
/// internal cursor every time [`Self::write`] is called, so custom `struct` implementations do
/// not need to manage slice offsets manually.
///
/// # Example
///
/// ```
/// use ssa_workflow::cache::{CanonicalEncode, CanonicalEncodeWriter};
///
/// struct Params {
///     rate: f64,
///     grid: [u16; 2],
/// }
///
/// impl CanonicalEncode for Params {
///     const SIZE: usize = f64::SIZE + <[u16; 2]>::SIZE;
///
///     unsafe fn encode_into(&self, buffer: &mut [u8]) {
///         CanonicalEncodeWriter::for_type::<Self>(buffer)
///             .write(&self.rate)
///             .write(&self.grid)
///             .finish();
///     }
/// }
/// ```
#[derive(Debug)]
pub struct CanonicalEncodeWriter<'a> {
    buffer: &'a mut [u8],
    offset: usize,
}

impl<'a> CanonicalEncodeWriter<'a> {
    /// Create a writer over the exact `T::SIZE` prefix of `buffer`.
    ///
    /// This is the usual entry point inside `impl CanonicalEncode for MyType`.
    #[inline]
    pub fn for_type<T: CanonicalEncode>(buffer: &'a mut [u8]) -> Self {
        let (prefix, _) = buffer.split_at_mut(T::SIZE);
        Self {
            buffer: prefix,
            offset: 0,
        }
    }

    /// Append one field using its [`CanonicalEncode`] implementation.
    #[inline]
    pub fn write<T: CanonicalEncode>(&mut self, value: &T) -> &mut Self {
        let end = self.offset + T::SIZE;
        let slot = &mut self.buffer[self.offset..end];
        unsafe { value.encode_into(slot) };
        self.offset = end;
        self
    }

    /// Finish encoding.
    ///
    /// In debug builds this asserts that the implementation wrote exactly `Self::SIZE` bytes.
    #[inline]
    pub fn finish(&mut self) {
        debug_assert!(
            self.offset == self.buffer.len(),
            "CanonicalEncodeWriter::finish called before filling the full buffer"
        );
    }
}

#[cfg(not(target_pointer_width = "64"))]
compile_error!("ssa-workflow supports only 64-bit targets");

macro_rules! impl_encode_for_unsigned {
    ($($t:path => $size:literal),+ $(,)?) => {
        $(
            impl CanonicalEncode for $t {
                const SIZE: usize = $size;

                #[inline]
                unsafe fn encode_into(&self, buffer: &mut [u8]) {
                    unsafe { buffer.get_unchecked_mut(..$size) }.copy_from_slice(&self.to_be_bytes());
                }
            }
        )+
    };
}

impl_encode_for_unsigned!(u8 => 1, u16 => 2, u32 => 4, u64 => 8, usize => 8, u128 => 16);

macro_rules! impl_encode_for_signed {
    ($($t:path => $size:literal),+ $(,)?) => {
        $(
            impl CanonicalEncode for $t {
                const SIZE: usize = $size;

                #[inline]
                unsafe fn encode_into(&self, buffer: &mut [u8]) {
                    let mut bytes = self.to_be_bytes();
                    // Flip the most-significant sign bit so that the byte-lexicographic order
                    // matches the natural numeric order: MIN encodes lowest, MAX encodes highest.
                    bytes[0] ^= 0x80;
                    unsafe { buffer.get_unchecked_mut(..$size) }.copy_from_slice(&bytes);
                }
            }
        )+
    };
}

impl_encode_for_signed!(i8 => 1, i16 => 2, i32 => 4, i64 => 8, isize => 8, i128 => 16);

macro_rules! impl_encode_for_float {
    ($($t:ident => $size:literal => $canonical_nan_bits:expr),+ $(,)?) => {
        $(
            // Grouped by IEEE 754 fields (sign, exponent, quiet bit, payload), not nibbles.
            #[allow(clippy::unusual_byte_groupings)]
            impl CanonicalEncode for $t {
                const SIZE: usize = $size;

                #[inline]
                unsafe fn encode_into(&self, buffer: &mut [u8]) {
                    // Normalize: all NaN payloads to one canonical positive quiet NaN,
                    // -0.0 to +0.0. We use an explicit bit pattern rather than `$t::NAN.to_bits()`
                    // because the Rust std docs state that NAN's sign and payload are arbitrary
                    // and may change across versions and targets.
                    let bits = if self.is_nan() {
                        $canonical_nan_bits
                    } else if *self == 0.0 {
                        0
                    } else {
                        self.to_bits()
                    };

                    // Order-preserving transformation:
                    // - Negative values (sign bit set): invert all bits.
                    // - Non-negative values (sign bit clear): flip only the sign bit.
                    //
                    // This yields: -inf < finite negatives < 0 < finite positives < +inf < NaN.
                    let sign_bit = 1 << ($size * 8 - 1);
                    let encoded = if bits & sign_bit != 0 {
                        !bits
                    } else {
                        bits ^ sign_bit
                    };

                    unsafe { buffer.get_unchecked_mut(..$size) }.copy_from_slice(&encoded.to_be_bytes());
                }
            }
        )+
    };
}

// Positive quiet NaN with zero payload (IEEE 754 field layout: sign_exponent_quiet_payload).
impl_encode_for_float!(
    f32 => 4 => 0b0_11111111_1_0000000000000000000000u32,
    f64 => 8 => 0b0_11111111111_1_000000000000000000000000000000000000000000000000000u64,
);

macro_rules! impl_encode_for_tuple {
    ($($T:ident $idx:tt),+) => {
        impl<$($T: CanonicalEncode),+> CanonicalEncode for ($($T,)+) {
            const SIZE: usize = 0 $(+ $T::SIZE)+;

            #[allow(unused_assignments)]
            unsafe fn encode_into(&self, buffer: &mut [u8]) {
                let mut offset = 0usize;
                $(
                    let end = offset + $T::SIZE;
                    unsafe {
                        self.$idx.encode_into(buffer.get_unchecked_mut(offset..end));
                    }
                    offset = end;
                )+
            }
        }
    };
}

impl_encode_for_tuple!(T0 0);
impl_encode_for_tuple!(T0 0, T1 1);
impl_encode_for_tuple!(T0 0, T1 1, T2 2);
impl_encode_for_tuple!(T0 0, T1 1, T2 2, T3 3);
impl_encode_for_tuple!(T0 0, T1 1, T2 2, T3 3, T4 4);
impl_encode_for_tuple!(T0 0, T1 1, T2 2, T3 3, T4 4, T5 5);
impl_encode_for_tuple!(T0 0, T1 1, T2 2, T3 3, T4 4, T5 5, T6 6);
impl_encode_for_tuple!(T0 0, T1 1, T2 2, T3 3, T4 4, T5 5, T6 6, T7 7);
impl_encode_for_tuple!(T0 0, T1 1, T2 2, T3 3, T4 4, T5 5, T6 6, T7 7, T8 8);
impl_encode_for_tuple!(T0 0, T1 1, T2 2, T3 3, T4 4, T5 5, T6 6, T7 7, T8 8, T9 9);
impl_encode_for_tuple!(T0 0, T1 1, T2 2, T3 3, T4 4, T5 5, T6 6, T7 7, T8 8, T9 9, T10 10);
impl_encode_for_tuple!(T0 0, T1 1, T2 2, T3 3, T4 4, T5 5, T6 6, T7 7, T8 8, T9 9, T10 10, T11 11);

impl<T: CanonicalEncode, const N: usize> CanonicalEncode for [T; N] {
    const SIZE: usize = T::SIZE * N;

    unsafe fn encode_into(&self, buf: &mut [u8]) {
        if T::SIZE == 0 {
            return;
        }

        let mut offset = 0usize;
        for item in self {
            let end = offset + T::SIZE;
            unsafe { item.encode_into(buf.get_unchecked_mut(offset..end)) };
            offset = end;
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::{CanonicalEncode, CanonicalEncodeWriter};

    fn canonical_encode_size<T: CanonicalEncode>(_: &T) -> usize {
        T::SIZE
    }

    fn encode<T: CanonicalEncode>(value: &T) -> Vec<u8> {
        let mut buffer = vec![0u8; T::SIZE];
        unsafe { value.encode_into(&mut buffer) };
        buffer
    }

    /// Assert that encoding `values` in order produces strictly increasing byte sequences.
    #[track_caller]
    fn assert_order_preserving<T: CanonicalEncode + std::fmt::Debug>(values: &[T]) {
        let encoded: Vec<Vec<u8>> = values.iter().map(encode).collect();
        for (i, window) in encoded.windows(2).enumerate() {
            assert!(
                window[0] < window[1],
                "order violated at index {i}: {:?} !< {:?}",
                values[i],
                values[i + 1]
            );
        }
    }

    macro_rules! assert_size {
        ($t:ty, $size:literal) => {
            assert_eq!(<$t as CanonicalEncode>::SIZE, $size);
        };
    }

    macro_rules! assert_encode {
        ($value:expr, [$($byte:literal),* $(,)?]) => {{
            let value = $value;
            let expected: &[u8] = &[$($byte),*];
            assert_eq!(expected.len(), canonical_encode_size(&value));

            let mut buffer = vec![0u8; expected.len()];
            let encoded = unsafe { value.encode_with_buffer(&mut buffer) };
            assert_eq!(encoded, expected);
        }};
    }

    mod unit_encode {
        use super::*;

        #[test]
        fn encodes_empty() {
            assert_encode!((), []);
        }
    }

    mod bool_encode {
        use super::*;

        #[test]
        fn encode_size() {
            assert_size!(bool, 1);
        }

        #[test]
        fn encodes_correctly() {
            assert_encode!(true, [0x01]);
            assert_encode!(false, [0x00]);
        }
    }

    mod int_encode {
        use super::*;

        #[test]
        fn encode_size() {
            assert_size!(u8, 1);
            assert_size!(u16, 2);
            assert_size!(u32, 4);
            assert_size!(u64, 8);
            assert_size!(usize, 8);
            assert_size!(u128, 16);
            assert_size!(i8, 1);
            assert_size!(i16, 2);
            assert_size!(i32, 4);
            assert_size!(i64, 8);
            assert_size!(isize, 8);
            assert_size!(i128, 16);
        }

        #[test]
        fn encodes_unsigned_big_endian() {
            assert_encode!(0x12u8, [0x12]);
            assert_encode!(0x0102u16, [0x01, 0x02]);
            assert_encode!(0x0102_0304u32, [0x01, 0x02, 0x03, 0x04]);
            assert_encode!(0x0102_0304_0506_0708u64, [
                0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
            ]);
            assert_encode!(0x0102_0304_0506_0708usize, [
                0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
            ]);
            assert_encode!(0x0102_0304_0506_0708_090a_0b0c_0d0e_0f10u128, [
                0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e,
                0x0f, 0x10,
            ]);
        }

        #[test]
        fn encodes_signed_order_preserving_golden_vectors() {
            // Signed encoding flips the MSB of the big-endian representation.
            // i8: -128 (0x80) -> 0x00, -1 (0xFF) -> 0x7F, 0 (0x00) -> 0x80, 127 (0x7F) -> 0xFF
            assert_encode!(i8::MIN, [0x00]);
            assert_encode!(-1i8, [0x7f]);
            assert_encode!(0i8, [0x80]);
            assert_encode!(1i8, [0x81]);
            assert_encode!(i8::MAX, [0xff]);

            // i16
            assert_encode!(i16::MIN, [0x00, 0x00]);
            assert_encode!(-1i16, [0x7f, 0xff]);
            assert_encode!(0i16, [0x80, 0x00]);
            assert_encode!(i16::MAX, [0xff, 0xff]);

            // i32
            assert_encode!(i32::MIN, [0x00, 0x00, 0x00, 0x00]);
            assert_encode!(-1i32, [0x7f, 0xff, 0xff, 0xff]);
            assert_encode!(0i32, [0x80, 0x00, 0x00, 0x00]);
            assert_encode!(i32::MAX, [0xff, 0xff, 0xff, 0xff]);

            // i64
            assert_encode!(i64::MIN, [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
            assert_encode!(-1i64, [0x7f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff]);
            assert_encode!(0i64, [0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
            assert_encode!(i64::MAX, [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff]);

            // isize (64-bit)
            assert_encode!(isize::MIN, [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
            assert_encode!(-1isize, [0x7f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff]);
            assert_encode!(0isize, [0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
            assert_encode!(isize::MAX, [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff]);

            // i128
            assert_encode!(i128::MIN, [
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00,
            ]);
            assert_encode!(-1i128, [
                0x7f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                0xff, 0xff,
            ]);
            assert_encode!(0i128, [
                0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00,
            ]);
            assert_encode!(i128::MAX, [
                0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                0xff, 0xff,
            ]);
        }

        #[test]
        fn signed_i8_encoding_preserves_numeric_order() {
            assert_order_preserving(&[
                i8::MIN,
                i8::MIN + 1,
                -100,
                -2,
                -1,
                0,
                1,
                2,
                100,
                i8::MAX - 1,
                i8::MAX,
            ]);
        }

        #[test]
        fn signed_i16_encoding_preserves_numeric_order() {
            assert_order_preserving(&[
                i16::MIN,
                i16::MIN + 1,
                -1000,
                -2,
                -1,
                0,
                1,
                2,
                1000,
                i16::MAX - 1,
                i16::MAX,
            ]);
        }

        #[test]
        fn signed_i32_encoding_preserves_numeric_order() {
            assert_order_preserving(&[
                i32::MIN,
                i32::MIN + 1,
                -100_000,
                -2,
                -1,
                0,
                1,
                2,
                100_000,
                i32::MAX - 1,
                i32::MAX,
            ]);
        }

        #[test]
        fn signed_i64_encoding_preserves_numeric_order() {
            assert_order_preserving(&[
                i64::MIN,
                i64::MIN + 1,
                -1_000_000_000,
                -2,
                -1,
                0,
                1,
                2,
                1_000_000_000,
                i64::MAX - 1,
                i64::MAX,
            ]);
        }

        #[test]
        fn signed_i128_encoding_preserves_numeric_order() {
            assert_order_preserving(&[
                i128::MIN,
                i128::MIN + 1,
                -1_000_000_000_000_000_000,
                -2,
                -1,
                0,
                1,
                2,
                1_000_000_000_000_000_000,
                i128::MAX - 1,
                i128::MAX,
            ]);
        }

        #[test]
        fn unsigned_encoding_preserves_numeric_order() {
            assert_order_preserving(&[0u64, 1, 2, 255, 256, u64::MAX - 1, u64::MAX]);
        }
    }

    mod float_encode {
        use super::*;

        #[test]
        fn encode_size() {
            assert_size!(f32, 4);
            assert_size!(f64, 8);
        }

        #[test]
        fn encodes_f32_order_preserving_golden_vectors() {
            // -inf: bits=0xFF800000, negative -> invert all -> 0x007FFFFF
            assert_encode!(f32::NEG_INFINITY, [0x00, 0x7f, 0xff, 0xff]);
            // -1.5: bits=0xBFC00000, negative -> invert all -> 0x403FFFFF
            assert_encode!(-1.5f32, [0x40, 0x3f, 0xff, 0xff]);
            // -0.0 normalized to +0.0: bits=0, non-negative -> flip sign -> 0x80000000
            assert_encode!(-0.0f32, [0x80, 0x00, 0x00, 0x00]);
            // +0.0: bits=0, non-negative -> flip sign -> 0x80000000
            assert_encode!(0.0f32, [0x80, 0x00, 0x00, 0x00]);
            // 1.5: bits=0x3FC00000, non-negative -> flip sign -> 0xBFC00000
            assert_encode!(1.5f32, [0xbf, 0xc0, 0x00, 0x00]);
            // +inf: bits=0x7F800000, non-negative -> flip sign -> 0xFF800000
            assert_encode!(f32::INFINITY, [0xff, 0x80, 0x00, 0x00]);
            // NaN: canonical bits=0x7FC00000, non-negative -> flip sign -> 0xFFC00000
            assert_encode!(f32::NAN, [0xff, 0xc0, 0x00, 0x00]);
            assert_encode!(f32::from_bits(0x7fc0_0001), [0xff, 0xc0, 0x00, 0x00]);
        }

        #[test]
        fn encodes_f64_order_preserving_golden_vectors() {
            // -inf: bits=0xFFF0000000000000, negative -> invert all -> 0x000FFFFFFFFFFFFF
            assert_encode!(f64::NEG_INFINITY, [
                0x00, 0x0f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            ]);
            // -1.5: bits=0xBFF8000000000000, negative -> invert all -> 0x4007FFFFFFFFFFFF
            assert_encode!(-1.5f64, [0x40, 0x07, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff]);
            // -0.0 normalized to +0.0: bits=0, non-negative -> flip sign -> 0x8000000000000000
            assert_encode!(-0.0f64, [0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
            // +0.0: bits=0, non-negative -> flip sign -> 0x8000000000000000
            assert_encode!(0.0f64, [0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
            // 1.5: bits=0x3FF8000000000000, non-negative -> flip sign -> 0xBFF8000000000000
            assert_encode!(1.5f64, [0xbf, 0xf8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
            // +inf: bits=0x7FF0000000000000, non-negative -> flip sign -> 0xFFF0000000000000
            assert_encode!(f64::INFINITY, [
                0xff, 0xf0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
            ]);
            // NaN: canonical bits=0x7FF8000000000000, non-negative -> flip sign ->
            // 0xFFF8000000000000
            assert_encode!(f64::NAN, [0xff, 0xf8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
            assert_encode!(f64::from_bits(0x7ff8_0000_0000_0001), [
                0xff, 0xf8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            ]);
        }

        #[test]
        fn f32_encoding_preserves_canonical_order() {
            let smallest_subnormal = f32::from_bits(1);
            assert_order_preserving(&[
                f32::NEG_INFINITY,
                -1e30,
                -1.5,
                -f32::MIN_POSITIVE,
                -smallest_subnormal,
                0.0,
                smallest_subnormal,
                f32::MIN_POSITIVE,
                1.5,
                1e30,
                f32::INFINITY,
                f32::NAN,
            ]);
        }

        #[test]
        fn f64_encoding_preserves_canonical_order() {
            let smallest_subnormal = f64::from_bits(1);
            assert_order_preserving(&[
                f64::NEG_INFINITY,
                -1e300,
                -1.5,
                -f64::MIN_POSITIVE,
                -smallest_subnormal,
                0.0,
                smallest_subnormal,
                f64::MIN_POSITIVE,
                1.5,
                1e300,
                f64::INFINITY,
                f64::NAN,
            ]);
        }

        #[test]
        fn negative_zero_and_positive_zero_encode_identically() {
            assert_eq!(encode(&-0.0f32), encode(&0.0f32));
            assert_eq!(encode(&-0.0f64), encode(&0.0f64));
        }

        #[test]
        fn all_nan_payloads_encode_identically() {
            // Positive quiet NaNs with different payloads.
            let nan_a = f32::from_bits(0x7fc0_0001);
            let nan_b = f32::from_bits(0x7fc0_dead);
            assert_eq!(encode(&nan_a), encode(&nan_b));
            assert_eq!(encode(&nan_a), encode(&f32::NAN));

            let nan_c = f64::from_bits(0x7ff8_0000_0000_0001);
            let nan_d = f64::from_bits(0x7ff8_dead_beef_0000);
            assert_eq!(encode(&nan_c), encode(&nan_d));
            assert_eq!(encode(&nan_c), encode(&f64::NAN));

            // Signaling NaNs (exponent all-ones, mantissa nonzero but quiet bit clear).
            let snan_f32 = f32::from_bits(0x7f80_0001);
            assert_eq!(encode(&snan_f32), encode(&f32::NAN));

            let snan_f64 = f64::from_bits(0x7ff0_0000_0000_0001);
            assert_eq!(encode(&snan_f64), encode(&f64::NAN));

            // Negative NaNs (sign bit set).
            let neg_nan_f32 = f32::from_bits(0xffc0_0001);
            assert_eq!(encode(&neg_nan_f32), encode(&f32::NAN));

            let neg_nan_f64 = f64::from_bits(0xfff8_0000_0000_0001);
            assert_eq!(encode(&neg_nan_f64), encode(&f64::NAN));
        }
    }

    mod tuple_encode {
        use super::*;

        #[test]
        fn concatenates_same_width_elements() {
            assert_encode!((0x0102_0304_0506_0708u64, 0x1112_1314_1516_1718u64), [
                0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16,
                0x17, 0x18,
            ]);
        }

        #[test]
        fn concatenates_mixed_width_elements() {
            assert_encode!((0x0102u16, 0x0304_0506_0708_090au64), [
                0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a,
            ]);
        }

        #[test]
        fn tuple_inherits_lexicographic_order_from_signed_components() {
            assert_order_preserving(&[
                (i32::MIN, 0),
                (-1, 0),
                (-1, i32::MAX),
                (0, i32::MIN),
                (0, 0),
                (0, 1),
                (1, i32::MIN),
                (i32::MAX, i32::MAX),
            ]);
        }

        #[test]
        fn tuple_inherits_lexicographic_order_from_float_components() {
            assert_order_preserving(&[
                (f64::NEG_INFINITY, 0u32),
                (-1.0, 0),
                (-1.0, u32::MAX),
                (0.0, 0),
                (0.0, 1),
                (1.0, 0),
                (f64::INFINITY, 0),
            ]);
        }
    }

    mod array_encode {
        use super::*;

        #[test]
        fn concatenates_elements() {
            assert_encode!([0x0102u16, 0x0304u16, 0x0506u16], [
                0x01, 0x02, 0x03, 0x04, 0x05, 0x06,
            ]);
        }

        #[test]
        fn zero_sized_array_is_empty() {
            assert_encode!([(); 8], []);
        }

        #[test]
        fn array_inherits_lexicographic_order_from_signed_elements() {
            assert_order_preserving(&[
                [i16::MIN, 0, 0],
                [-1, 0, 0],
                [-1, i16::MAX, i16::MAX],
                [0, i16::MIN, 0],
                [0, 0, 0],
                [0, 0, 1],
                [1, i16::MIN, i16::MIN],
                [i16::MAX, i16::MAX, i16::MAX],
            ]);
        }
    }

    mod writer_encode {
        use super::*;

        struct SearchKey {
            generation: u64,
            selection: f64,
            counts: [u16; 3],
        }

        impl CanonicalEncode for SearchKey {
            const SIZE: usize = u64::SIZE + f64::SIZE + <[u16; 3]>::SIZE;

            unsafe fn encode_into(&self, buffer: &mut [u8]) {
                CanonicalEncodeWriter::for_type::<Self>(buffer)
                    .write(&self.generation)
                    .write(&self.selection)
                    .write(&self.counts)
                    .finish();
            }
        }

        #[test]
        fn encodes_custom_struct_fields_in_order() {
            assert_encode!(
                SearchKey {
                    generation: 0x0102_0304_0506_0708,
                    selection: -0.0,
                    counts: [0x090a, 0x0b0c, 0x0d0e],
                },
                [
                    0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, // generation
                    0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, // selection: -0.0 canonicalized to +0.0, sign bit flipped
                    0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, // counts
                ]
            );
        }

        #[test]
        #[cfg(debug_assertions)]
        #[should_panic(
            expected = "CanonicalEncodeWriter::finish called before filling the full buffer"
        )]
        fn finish_panics_when_bytes_remain() {
            let mut buffer = [0u8; SearchKey::SIZE];
            let mut writer = CanonicalEncodeWriter::for_type::<SearchKey>(&mut buffer);
            writer.write(&1u64);
            writer.finish();
        }
    }
}
