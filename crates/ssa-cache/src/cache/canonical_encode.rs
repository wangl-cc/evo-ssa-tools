/// Encode a key into canonical bytes.
///
/// Implementations should be deterministic and consistent across different builds and runs, and
/// always have a consistent size `Self::SIZE`.
///
/// # Implementations
///
/// - For integers, the encoding is big-endian;
/// - For floats, NaN values are normalized to a canonical NaN, and -0.0 is treated as +0.0.
/// - For tuples and arrays, the encoding is the concatenation of the encodings of each element.
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

impl_encode_for_int!(u8 => 1, u16 => 2, u32 => 4, u64 => 8, u128 => 16);
impl_encode_for_int!(i8 => 1, i16 => 2, i32 => 4, i64 => 8, i128 => 16);

#[cfg(target_pointer_width = "32")]
impl_encode_for_int!(usize => 4, isize => 4);

#[cfg(target_pointer_width = "64")]
impl_encode_for_int!(usize => 8, isize => 8);

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
mod tests {
    use super::CanonicalEncode;

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
        let encoded_nan_non_canonical = unsafe { nan_non_canonical.encode_with_buffer(&mut buffer_a) };
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
}
