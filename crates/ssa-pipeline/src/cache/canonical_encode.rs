/// Encode a key into canonical bytes.
///
/// Implementations should be deterministic and consistent across different builds and runs, and
/// always have a consistent size `Self::SIZE`.
///
/// # Portability
///
/// `ssa-pipeline` targets 64-bit platforms only. This avoids platform-dependent key encodings
/// (e.g.
/// `usize` width) and makes cache keys stable across builds.
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
    /// The buffer must be with length at least `Self::SIZE`.
    /// And implementation should only access buffer[..Self::SIZE].
    unsafe fn encode_into(&self, buffer: &mut [u8]);

    /// Encode self into the provided buffer and return the encoded bytes.
    ///
    /// This is a convenience wrapper around `encode_into` that returns the encoded bytes.
    ///
    /// ## Safety
    ///
    /// The buffer must be with length at least `Self::SIZE`.
    /// And implementation should only access buffer[..Self::SIZE].
    unsafe fn encode_with_buffer<'b>(&self, buffer: &'b mut [u8]) -> &'b [u8] {
        unsafe { self.encode_into(buffer) };
        &buffer[..Self::SIZE]
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
/// use ssa_pipeline::cache::{CanonicalEncode, CanonicalEncodeWriter};
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
                    buffer[..$size].copy_from_slice(&self.to_be_bytes());
                }
            }
        )+
    };
}

impl_encode_for_int!(u8 => 1, u16 => 2, u32 => 4, u64 => 8, usize => 8, u128 => 16);
impl_encode_for_int!(i8 => 1, i16 => 2, i32 => 4, i64 => 8, isize => 8, i128 => 16);

#[cfg(not(target_pointer_width = "64"))]
compile_error!("ssa-pipeline supports only 64-bit targets");

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
                    buffer[..$size].copy_from_slice(&bits.to_be_bytes());
                }
            }
        )+
    };
}

impl_encode_for_float!(f32 => 4, f64 => 8);

impl<T1: CanonicalEncode, T2: CanonicalEncode> CanonicalEncode for (T1, T2) {
    const SIZE: usize = T1::SIZE + T2::SIZE;

    unsafe fn encode_into(&self, buffer: &mut [u8]) {
        unsafe {
            self.0.encode_into(&mut buffer[..T1::SIZE]);
            self.1.encode_into(&mut buffer[T1::SIZE..Self::SIZE]);
        }
    }
}

impl<T: CanonicalEncode, const N: usize> CanonicalEncode for [T; N] {
    const SIZE: usize = T::SIZE * N;

    unsafe fn encode_into(&self, buf: &mut [u8]) {
        for (item, chunk) in self.iter().zip(buf.chunks_exact_mut(T::SIZE)) {
            unsafe { item.encode_into(chunk) }
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::{CanonicalEncode, CanonicalEncodeWriter};

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
    fn test_float_encode_size_matches_type_width() {
        assert_eq!(f32::SIZE, 4);
        assert_eq!(f64::SIZE, 8);
    }

    #[test]
    fn test_tuple_encode_concatenates_same_width_elements() {
        let value = (0x0102_0304_0506_0708u64, 0x1112_1314_1516_1718u64);
        let mut buffer = vec![0u8; <(u64, u64) as CanonicalEncode>::SIZE];
        let encoded = unsafe { value.encode_with_buffer(&mut buffer) };

        let mut expected = Vec::new();
        expected.extend_from_slice(&0x0102_0304_0506_0708u64.to_be_bytes());
        expected.extend_from_slice(&0x1112_1314_1516_1718u64.to_be_bytes());
        assert_eq!(encoded, expected);
    }

    #[test]
    fn test_tuple_encode_concatenates_mixed_width_elements() {
        let value = (0x0102u16, 0x0304_0506_0708_090au64);
        let mut buffer = vec![0u8; <(u16, u64) as CanonicalEncode>::SIZE];
        let encoded = unsafe { value.encode_with_buffer(&mut buffer) };

        let mut expected = Vec::new();
        expected.extend_from_slice(&0x0102u16.to_be_bytes());
        expected.extend_from_slice(&0x0304_0506_0708_090au64.to_be_bytes());
        assert_eq!(encoded, expected);
    }

    #[test]
    fn test_float_encode_canonicalization() {
        let mut buffer_a = vec![0u8; f32::SIZE];
        let mut buffer_b = vec![0u8; f32::SIZE];
        let mut buffer_c = vec![0u8; f32::SIZE];

        let nan_non_canonical = f32::from_bits(0x7fc0_0001);
        let nan_canonical = f32::NAN;
        let encoded_nan_non_canonical =
            unsafe { nan_non_canonical.encode_with_buffer(&mut buffer_a) };
        let encoded_nan_canonical = unsafe { nan_canonical.encode_with_buffer(&mut buffer_b) };
        assert_eq!(encoded_nan_non_canonical, encoded_nan_canonical);

        let encoded_pos_zero = unsafe { 0.0f32.encode_with_buffer(&mut buffer_c) };
        let mut buffer_d = vec![0u8; f32::SIZE];
        let encoded_neg_zero = unsafe { (-0.0f32).encode_with_buffer(&mut buffer_d) };
        assert_eq!(encoded_pos_zero, encoded_neg_zero);

        let normal = 1.5f32;
        let mut buffer_e = vec![0u8; f32::SIZE];
        let encoded_normal = unsafe { normal.encode_with_buffer(&mut buffer_e) };
        assert_eq!(encoded_normal, normal.to_bits().to_be_bytes());
    }

    #[test]
    fn test_array_encode_concatenates_elements() {
        let value = [0x0102u16, 0x0304u16, 0x0506u16];
        let mut buffer = vec![0u8; <[u16; 3] as CanonicalEncode>::SIZE];
        let encoded = unsafe { value.encode_with_buffer(&mut buffer) };

        let mut expected = Vec::new();
        expected.extend_from_slice(&0x0102u16.to_be_bytes());
        expected.extend_from_slice(&0x0304u16.to_be_bytes());
        expected.extend_from_slice(&0x0506u16.to_be_bytes());
        assert_eq!(encoded, expected);
    }

    #[test]
    fn test_writer_encodes_custom_struct_fields_in_order() {
        let value = SearchKey {
            generation: 0x0102_0304_0506_0708,
            selection: -0.0,
            counts: [0x090a, 0x0b0c, 0x0d0e],
        };
        let mut buffer = vec![0u8; SearchKey::SIZE];
        let encoded = unsafe { value.encode_with_buffer(&mut buffer) };

        let mut expected = Vec::new();
        expected.extend_from_slice(&value.generation.to_be_bytes());
        expected.extend_from_slice(&0f64.to_bits().to_be_bytes());
        expected.extend_from_slice(&0x090au16.to_be_bytes());
        expected.extend_from_slice(&0x0b0cu16.to_be_bytes());
        expected.extend_from_slice(&0x0d0eu16.to_be_bytes());
        assert_eq!(encoded, expected);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(
        expected = "CanonicalEncodeWriter::finish called before filling the full buffer"
    )]
    fn test_writer_finish_panics_when_bytes_remain() {
        let mut buffer = [0u8; SearchKey::SIZE];
        let mut writer = CanonicalEncodeWriter::for_type::<SearchKey>(&mut buffer);
        writer.write(&1u64);
        writer.finish();
    }
}
