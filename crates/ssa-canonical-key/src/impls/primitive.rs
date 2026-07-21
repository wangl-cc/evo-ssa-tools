use crate::{CanonicalEncode, CanonicalWriter};

impl CanonicalEncode for bool {
    const SIZE: usize = 1;

    #[inline]
    fn encode(&self, writer: &mut CanonicalWriter<'_>) {
        writer.write_bytes(&[*self as u8]);
    }
}

#[cfg(not(target_pointer_width = "64"))]
compile_error!("ssa-canonical-key supports only 64-bit targets");

macro_rules! impl_encode_for_unsigned {
    ($($t:path => $size:literal),+ $(,)?) => {
        $(
            impl CanonicalEncode for $t {
                const SIZE: usize = $size;

                #[inline]
                fn encode(&self, writer: &mut CanonicalWriter<'_>) {
                    writer.write_bytes(&self.to_be_bytes());
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
                fn encode(&self, writer: &mut CanonicalWriter<'_>) {
                    let mut bytes = self.to_be_bytes();
                    // Flip the most-significant sign bit so that the byte-lexicographic order
                    // matches the natural numeric order: MIN encodes lowest, MAX encodes highest.
                    bytes[0] ^= 0x80;
                    writer.write_bytes(&bytes);
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
                fn encode(&self, writer: &mut CanonicalWriter<'_>) {
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

                    writer.write_bytes(&encoded.to_be_bytes());
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

#[cfg(test)]
mod tests {
    mod boolean {
        use crate::impls::test_support::*;

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

    mod int {
        mod unsigned {
            use crate::impls::test_support::*;

            #[test]
            fn encode_size() {
                assert_size!(u8, 1);
                assert_size!(u16, 2);
                assert_size!(u32, 4);
                assert_size!(u64, 8);
                assert_size!(usize, 8);
                assert_size!(u128, 16);
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
                    0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
                    0x0e, 0x0f, 0x10,
                ]);
            }

            #[test]
            fn unsigned_encoding_preserves_numeric_order() {
                assert_order_preserving(&[0u64, 1, 2, 255, 256, u64::MAX - 1, u64::MAX]);
            }
        }

        mod signed {
            use crate::impls::test_support::*;

            #[test]
            fn encode_size() {
                assert_size!(i8, 1);
                assert_size!(i16, 2);
                assert_size!(i32, 4);
                assert_size!(i64, 8);
                assert_size!(isize, 8);
                assert_size!(i128, 16);
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
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00,
                ]);
                assert_encode!(-1i128, [
                    0x7f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                    0xff, 0xff, 0xff,
                ]);
                assert_encode!(0i128, [
                    0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00,
                ]);
                assert_encode!(i128::MAX, [
                    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                    0xff, 0xff, 0xff,
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
        }
    }

    mod float {
        use crate::impls::test_support::*;

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
}
