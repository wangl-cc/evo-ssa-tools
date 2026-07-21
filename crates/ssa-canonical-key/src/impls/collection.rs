use crate::{CanonicalEncode, CanonicalWriter};

impl<T: CanonicalEncode, const N: usize> CanonicalEncode for [T; N] {
    const SIZE: usize = T::SIZE * N;

    fn encode(&self, writer: &mut CanonicalWriter<'_>) {
        for item in self {
            writer.write(item);
        }
    }
}

impl CanonicalEncode for () {
    const SIZE: usize = 0;

    #[inline]
    fn encode(&self, _: &mut CanonicalWriter<'_>) {}
}

macro_rules! impl_encode_for_tuple {
    ($($T:ident $idx:tt),*) => {
        impl<$($T: CanonicalEncode),*> CanonicalEncode for ($($T,)*) {
            const SIZE: usize = 0 $(+ $T::SIZE)*;

            fn encode(&self, writer: &mut CanonicalWriter<'_>) {
                $(
                    writer.write(&self.$idx);
                )*
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

#[cfg(test)]
mod tests {
    mod array {
        use crate::impls::test_support::*;

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

    mod tuple {
        use crate::impls::test_support::*;

        #[test]
        fn empty_tuple_encodes_empty() {
            assert_encode!((), []);
        }

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
}
