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
/// # Implementations
///
/// - For integers, the encoding is big-endian;
/// - For floats, NaN values are normalized to a canonical NaN, and -0.0 is treated as +0.0.
/// - For tuples and arrays, the encoding is the concatenation of the encodings of each element.
/// - For custom structs, [`CanonicalEncodeWriter`] can write each field in sequence.
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

macro_rules! impl_encode_for_int {
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

impl_encode_for_int!(u8 => 1, u16 => 2, u32 => 4, u64 => 8, usize => 8, u128 => 16);
impl_encode_for_int!(i8 => 1, i16 => 2, i32 => 4, i64 => 8, isize => 8, i128 => 16);

#[cfg(not(target_pointer_width = "64"))]
compile_error!("ssa-workflow supports only 64-bit targets");

macro_rules! impl_encode_for_float {
    ($($t:ident => $size:literal),+ $(,)?) => {
        $(
            impl CanonicalEncode for $t {
                const SIZE: usize = $size;

                #[inline]
                unsafe fn encode_into(&self, buffer: &mut [u8]) {
                    let bits = if self.is_nan() {
                        $t::NAN.to_bits()
                    } else if *self == 0.0 {
                        0
                    } else {
                        self.to_bits()
                    };
                    unsafe { buffer.get_unchecked_mut(..$size) }.copy_from_slice(&bits.to_be_bytes());
                }
            }
        )+
    };
}

impl_encode_for_float!(f32 => 4, f64 => 8);

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
        fn encodes_signed_big_endian() {
            assert_encode!(-0x12i8, [0xee]);
            assert_encode!(-0x0102i16, [0xfe, 0xfe]);
            assert_encode!(-0x0102_0304i32, [0xfe, 0xfd, 0xfc, 0xfc]);
            assert_encode!(-0x0102_0304_0506_0708i64, [
                0xfe, 0xfd, 0xfc, 0xfb, 0xfa, 0xf9, 0xf8, 0xf8,
            ]);
            assert_encode!(-0x0102_0304_0506_0708isize, [
                0xfe, 0xfd, 0xfc, 0xfb, 0xfa, 0xf9, 0xf8, 0xf8,
            ]);
            assert_encode!(-0x0102_0304_0506_0708_090a_0b0c_0d0e_0f10i128, [
                0xfe, 0xfd, 0xfc, 0xfb, 0xfa, 0xf9, 0xf8, 0xf7, 0xf6, 0xf5, 0xf4, 0xf3, 0xf2, 0xf1,
                0xf0, 0xf0,
            ]);
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
        fn encodes_f32_canonical_bytes() {
            assert_encode!(f32::from_bits(0x7fc0_0001), [0x7f, 0xc0, 0x00, 0x00]);
            assert_encode!(f32::NAN, [0x7f, 0xc0, 0x00, 0x00]);
            assert_encode!(0.0f32, [0x00, 0x00, 0x00, 0x00]);
            assert_encode!(-0.0f32, [0x00, 0x00, 0x00, 0x00]);
            assert_encode!(1.5f32, [0x3f, 0xc0, 0x00, 0x00]);
        }

        #[test]
        fn encodes_f64_canonical_bytes() {
            assert_encode!(f64::from_bits(0x7ff8_0000_0000_0001), [
                0x7f, 0xf8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            ]);
            assert_encode!(f64::NAN, [0x7f, 0xf8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,]);
            assert_encode!(0.0f64, [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,]);
            assert_encode!(-0.0f64, [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,]);
            assert_encode!(1.5f64, [0x3f, 0xf8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,]);
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
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, // selection: -0.0 canonicalized to +0.0
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
