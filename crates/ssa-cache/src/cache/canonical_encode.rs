/// Encode a canonical key into bytes.
///
/// Implementations should be deterministic and stable for cache keys.
pub trait CanonicalEncode {
    const SIZE: usize;

    /// Encode self
    ///
    /// ## Safety
    ///
    /// The buffer must be with length at least `Self::SIZE`.
    /// And implementation should only access buffer[..Self::SIZE].
    unsafe fn encode_into(&self, buffer: &mut [u8]);

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
    ($($t:path => $nan:path => $size:literal),+ $(,)?) => {
        $(
            impl CanonicalEncode for $t {
                const SIZE: usize = 8;

                #[inline]
                unsafe fn encode_into(&self, buffer: &mut [u8]) {
                    let bits = if self.is_nan() {
                        $nan.to_bits()
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

impl_encode_for_float!(f32 => f32::NAN => 4, f64 => f64::NAN => 8);

impl<T1: CanonicalEncode, T2: CanonicalEncode> CanonicalEncode for (T1, T2) {
    const SIZE: usize = T1::SIZE + T2::SIZE;

    unsafe fn encode_into(&self, buffer: &mut [u8]) {
        unsafe {
            self.0.encode_into(&mut buffer[..T1::SIZE]);
            self.1.encode_into(&mut buffer[T1::SIZE..T2::SIZE]);
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
