/// Number of bytes used to prefix a cache key with its schema signature.
pub const SCHEMA_SIGNATURE_SIZE: usize = 4;

/// Compute a CRC32C schema signature.
///
/// This is intentionally small: the signature is a key-schema discriminator, not a globally unique
/// identifier or cryptographic hash.
pub const fn schema_signature(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    let mut i = 0usize;
    while i < bytes.len() {
        crc = crc32c_update_byte(crc, bytes[i]);
        i += 1;
    }
    !crc
}

/// Extend a schema signature with another 32-bit schema component.
pub const fn extend_schema_signature(signature: u32, value: u32) -> u32 {
    let mut crc = !signature;
    let bytes = value.to_be_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        crc = crc32c_update_byte(crc, bytes[i]);
        i += 1;
    }
    !crc
}

/// Extend a schema signature with a `usize` schema component.
pub const fn extend_schema_signature_usize(signature: u32, value: usize) -> u32 {
    let mut crc = !signature;
    let bytes = value.to_be_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        crc = crc32c_update_byte(crc, bytes[i]);
        i += 1;
    }
    !crc
}

const fn crc32c_update_byte(mut crc: u32, byte: u8) -> u32 {
    crc ^= byte as u32;
    let mut bit = 0;
    while bit < 8 {
        let mask = 0u32.wrapping_sub(crc & 1);
        crc = (crc >> 1) ^ (0x82f6_3b78 & mask);
        bit += 1;
    }
    crc
}

/// Encode a key payload into canonical bytes.
///
/// Implementations should be deterministic and consistent across different builds and runs, and
/// must always write exactly `Self::SIZE` bytes.
///
/// # Schema signature
///
/// `SCHEMA_SIGNATURE` is written before the payload when a value is used as a cache key. It is not
/// required to be globally unique, but it must change whenever this type's canonical encoding
/// changes incompatibly within a cache keyspace. This includes field order, field meaning, field
/// type, and primitive encoding policy changes.
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
///
/// # Safety
///
/// Manual implementations are unsafe because the cache system relies on `SIZE`,
/// `SCHEMA_SIGNATURE`, and `encode_into` being mutually consistent. Incorrect implementations can
/// cause cache misses, stale cache reuse, or buffer access outside the documented payload range.
pub unsafe trait CanonicalEncode {
    /// Number of payload bytes written by [`Self::encode_into`].
    const SIZE: usize;
    /// Number of bytes written by [`Self::encode_key_with_buffer`].
    const KEY_SIZE: usize = SCHEMA_SIGNATURE_SIZE + Self::SIZE;
    /// Stable discriminator for this type's key encoding schema.
    const SCHEMA_SIGNATURE: u32;

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
        &buffer[..Self::SIZE]
    }

    /// Encode self as a cache key, prefixing the payload with [`Self::SCHEMA_SIGNATURE`].
    ///
    /// ## Safety
    ///
    /// The buffer must have length at least `Self::KEY_SIZE`.
    /// Implementations must only access `buffer[..Self::KEY_SIZE]`.
    unsafe fn encode_key_with_buffer<'b>(&self, buffer: &'b mut [u8]) -> &'b [u8] {
        buffer[..SCHEMA_SIGNATURE_SIZE].copy_from_slice(&Self::SCHEMA_SIGNATURE.to_be_bytes());
        unsafe { self.encode_into(&mut buffer[SCHEMA_SIGNATURE_SIZE..Self::KEY_SIZE]) };
        &buffer[..Self::KEY_SIZE]
    }
}

unsafe impl CanonicalEncode for () {
    const SCHEMA_SIGNATURE: u32 = schema_signature(b"ssa-workflow:canonical-encode:v1;unit");
    const SIZE: usize = 0;

    #[inline]
    unsafe fn encode_into(&self, _buffer: &mut [u8]) {}
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
/// unsafe impl CanonicalEncode for Params {
///     const SCHEMA_SIGNATURE: u32 = 1;
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
    ($($t:path => ($size:literal, $schema:literal)),+ $(,)?) => {
        $(
            unsafe impl CanonicalEncode for $t {
                const SIZE: usize = $size;
                const SCHEMA_SIGNATURE: u32 = schema_signature($schema.as_bytes());

                #[inline]
                unsafe fn encode_into(&self, buffer: &mut [u8]) {
                    buffer[..$size].copy_from_slice(&self.to_be_bytes());
                }
            }
        )+
    };
}

impl_encode_for_int!(
    u8 => (1, "ssa-workflow:canonical-encode:v1;u8-be"),
    u16 => (2, "ssa-workflow:canonical-encode:v1;u16-be"),
    u32 => (4, "ssa-workflow:canonical-encode:v1;u32-be"),
    u64 => (8, "ssa-workflow:canonical-encode:v1;u64-be"),
    usize => (8, "ssa-workflow:canonical-encode:v1;usize64-be"),
    u128 => (16, "ssa-workflow:canonical-encode:v1;u128-be"),
);
impl_encode_for_int!(
    i8 => (1, "ssa-workflow:canonical-encode:v1;i8-be"),
    i16 => (2, "ssa-workflow:canonical-encode:v1;i16-be"),
    i32 => (4, "ssa-workflow:canonical-encode:v1;i32-be"),
    i64 => (8, "ssa-workflow:canonical-encode:v1;i64-be"),
    isize => (8, "ssa-workflow:canonical-encode:v1;isize64-be"),
    i128 => (16, "ssa-workflow:canonical-encode:v1;i128-be"),
);

#[cfg(not(target_pointer_width = "64"))]
compile_error!("ssa-workflow supports only 64-bit targets");

macro_rules! impl_encode_for_float {
    ($($t:ident => ($size:literal, $schema:literal)),+ $(,)?) => {
        $(
            unsafe impl CanonicalEncode for $t {
                const SIZE: usize = $size;
                const SCHEMA_SIGNATURE: u32 = schema_signature($schema.as_bytes());

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

impl_encode_for_float!(
    f32 => (4, "ssa-workflow:canonical-encode:v1;f32-be-canonical-nan-pos-zero"),
    f64 => (8, "ssa-workflow:canonical-encode:v1;f64-be-canonical-nan-pos-zero"),
);

macro_rules! impl_encode_for_tuple {
    ($($T:ident $idx:tt),+) => {
        unsafe impl<$($T: CanonicalEncode),+> CanonicalEncode for ($($T,)+) {
            const SIZE: usize = 0 $(+ $T::SIZE)+;
            const SCHEMA_SIGNATURE: u32 = {
                let signature = schema_signature(b"ssa-workflow:canonical-encode:v1;tuple");
                $(
                    let signature = extend_schema_signature(signature, $T::SCHEMA_SIGNATURE);
                )+
                signature
            };

            #[allow(unused_assignments)]
            unsafe fn encode_into(&self, buffer: &mut [u8]) {
                let mut offset = 0usize;
                $(
                    unsafe {
                        self.$idx.encode_into(&mut buffer[offset..offset + $T::SIZE]);
                    }
                    offset += $T::SIZE;
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

unsafe impl<T: CanonicalEncode, const N: usize> CanonicalEncode for [T; N] {
    const SCHEMA_SIGNATURE: u32 = {
        let signature = schema_signature(b"ssa-workflow:canonical-encode:v1;array");
        let signature = extend_schema_signature(signature, T::SCHEMA_SIGNATURE);
        extend_schema_signature_usize(signature, N)
    };
    const SIZE: usize = T::SIZE * N;

    unsafe fn encode_into(&self, buf: &mut [u8]) {
        if T::SIZE == 0 {
            return;
        }

        for (item, chunk) in self.iter().zip(buf.chunks_exact_mut(T::SIZE)) {
            unsafe { item.encode_into(chunk) }
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::{
        CanonicalEncode, CanonicalEncodeWriter, extend_schema_signature,
        extend_schema_signature_usize, schema_signature,
    };

    struct SearchKey {
        generation: u64,
        selection: f64,
        counts: [u16; 3],
    }

    unsafe impl CanonicalEncode for SearchKey {
        const SCHEMA_SIGNATURE: u32 = 1;
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
    fn test_schema_signature_matches_crc32c() {
        let schema = b"ssa-workflow:canonical-encode:v1;test";
        assert_eq!(schema_signature(schema), crc32c::crc32c(schema));
    }

    #[test]
    fn test_extend_schema_signature_matches_crc32c() {
        let schema = b"ssa-workflow:canonical-encode:v1;test";
        let mut expected = schema.to_vec();
        expected.extend_from_slice(&0x0102_0304u32.to_be_bytes());
        expected.extend_from_slice(&5usize.to_be_bytes());

        let signature = schema_signature(schema);
        let signature = extend_schema_signature(signature, 0x0102_0304);
        let signature = extend_schema_signature_usize(signature, 5);

        assert_eq!(signature, crc32c::crc32c(&expected));
    }

    #[test]
    fn test_unit_encode_is_empty() {
        let mut buffer = [];
        let encoded = unsafe { ().encode_with_buffer(&mut buffer) };
        assert!(encoded.is_empty());
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
    fn test_zero_sized_array_encode_is_empty() {
        let value = [(); 8];
        let mut buffer = [];
        let encoded = unsafe { value.encode_with_buffer(&mut buffer) };

        assert!(encoded.is_empty());
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
    fn test_key_encode_prefixes_schema_signature() {
        let value = SearchKey {
            generation: 0x0102_0304_0506_0708,
            selection: -0.0,
            counts: [0x090a, 0x0b0c, 0x0d0e],
        };
        let mut payload_buffer = vec![0u8; SearchKey::SIZE];
        let payload = unsafe { value.encode_with_buffer(&mut payload_buffer) }.to_vec();

        let mut key_buffer = vec![0u8; SearchKey::KEY_SIZE];
        let key = unsafe { value.encode_key_with_buffer(&mut key_buffer) };

        let mut expected = SearchKey::SCHEMA_SIGNATURE.to_be_bytes().to_vec();
        expected.extend_from_slice(&payload);
        assert_eq!(key, expected);
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
