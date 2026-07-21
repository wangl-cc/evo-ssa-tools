#![doc = include_str!("../README.md")]
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

use std::{any::type_name, marker::PhantomData};

/// Version of the built-in canonical encodings.
///
/// Persistent stores should include this value in their namespace identity. A future change to
/// any built-in encoded bytes must also change this value. This constant does not version
/// user-defined [`CanonicalEncode`] implementations.
pub const BUILTIN_ENCODING_VERSION: u32 = 2;

/// Encode a computation input into canonical bytes.
///
/// Implementations should be deterministic and consistent across different builds and runs, and
/// must always write exactly `Self::SIZE` bytes.
///
/// # Portability
///
/// This crate targets 64-bit platforms only. This avoids platform-dependent encodings such as
/// `usize` width and makes canonical input bytes stable across builds.
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
///
/// An implementation defines the domain-specific equivalence represented by equal encoded bytes.
/// The built-in implementations express this crate's default policies. Domains that need different
/// behavior, such as preserving floating-point NaN payloads or the sign of zero, should use a
/// wrapper or newtype with its own implementation.
pub trait CanonicalEncode {
    /// Exact number of bytes written by [`Self::encode`].
    const SIZE: usize;

    /// Write this value's canonical bytes to `writer`.
    ///
    /// Implementations must write exactly [`Self::SIZE`] bytes. The writer enforces this contract
    /// before encoded bytes can be returned, including for nested values.
    fn encode(&self, writer: &mut CanonicalWriter<'_>);
}

/// Sequential writer for building a [`CanonicalEncode`] implementation field by field.
///
/// The writer owns the bounds checks and byte-count validation for the current value. Custom
/// implementations only describe their field order through [`Self::write`] or
/// [`Self::write_bytes`].
///
/// # Example
///
/// ```
/// use canonical_input_encoding::{CanonicalBuffer, CanonicalEncode, CanonicalWriter};
///
/// struct Params {
///     rate: f64,
///     grid: [u16; 2],
/// }
///
/// impl CanonicalEncode for Params {
///     const SIZE: usize = f64::SIZE + <[u16; 2]>::SIZE;
///
///     fn encode(&self, writer: &mut CanonicalWriter<'_>) {
///         writer.write(&self.rate).write(&self.grid);
///     }
/// }
///
/// let mut buffer = CanonicalBuffer::<Params>::new();
/// assert_eq!(
///     buffer
///         .encode(&Params {
///             rate: 0.5,
///             grid: [16, 32]
///         })
///         .len(),
///     Params::SIZE
/// );
/// ```
#[derive(Debug)]
pub struct CanonicalWriter<'a> {
    buffer: &'a mut [u8],
    position: usize,
}

impl<'a> CanonicalWriter<'a> {
    #[inline]
    fn new(buffer: &'a mut [u8]) -> Self {
        Self {
            buffer,
            position: 0,
        }
    }

    /// Append one field using its [`CanonicalEncode`] implementation.
    ///
    /// The nested implementation is validated before this method returns.
    ///
    /// # Panics
    ///
    /// Panics if the field exceeds the space remaining for the enclosing value, calculating the
    /// new write position overflows `usize`, or the field's implementation writes a number of
    /// bytes different from `T::SIZE`.
    #[inline]
    pub fn write<T: CanonicalEncode>(&mut self, value: &T) -> &mut Self {
        let end = self.checked_end(T::SIZE);
        let position = self.position;
        let capacity = self.buffer.len();
        let slot = self.buffer.get_mut(position..end).unwrap_or_else(|| {
            panic!(
                "canonical encoder wrote past its declared size: position={}, write={}, capacity={capacity}",
                position,
                T::SIZE,
            )
        });
        let mut child = CanonicalWriter {
            buffer: slot,
            position: 0,
        };
        value.encode(&mut child);
        child.finish::<T>();
        self.position = end;
        self
    }

    /// Append canonical bytes directly.
    ///
    /// This is intended for primitive encodings or domain types whose canonical representation is
    /// already available as bytes. It does not add framing or a length prefix.
    ///
    /// # Panics
    ///
    /// Panics if `bytes` exceeds the space remaining for the enclosing value, or if calculating
    /// the new write position overflows `usize`.
    #[inline]
    pub fn write_bytes(&mut self, bytes: &[u8]) -> &mut Self {
        let end = self.checked_end(bytes.len());
        let position = self.position;
        let capacity = self.buffer.len();
        let destination = self.buffer.get_mut(position..end).unwrap_or_else(|| {
            panic!(
                "canonical encoder wrote past its declared size: position={}, write={}, capacity={capacity}",
                position,
                bytes.len(),
            )
        });
        destination.copy_from_slice(bytes);
        self.position = end;
        self
    }

    #[inline]
    fn checked_end(&self, additional: usize) -> usize {
        self.position.checked_add(additional).unwrap_or_else(|| {
            panic!(
                "canonical encoding length overflow: position={}, write={additional}",
                self.position,
            )
        })
    }

    /// Validate that the current value wrote its exact declared width.
    ///
    /// This check is intentionally active in release builds so a malformed implementation cannot
    /// return a stale or ambiguous encoding.
    #[inline]
    fn finish<T: CanonicalEncode>(self) {
        assert_eq!(
            self.position,
            self.buffer.len(),
            "CanonicalEncode implementation for {} declared SIZE={}, but wrote {} bytes",
            type_name::<T>(),
            T::SIZE,
            self.position,
        );
    }
}

/// Owning, reusable buffer for one [`CanonicalEncode`] type.
///
/// Construction allocates exactly `T::SIZE` initialized bytes. Repeated calls to [`Self::encode`]
/// reuse that allocation, which makes this type suitable as per-worker state in schedulers.
#[derive(Debug)]
pub struct CanonicalBuffer<T: CanonicalEncode> {
    bytes: Box<[u8]>,
    marker: PhantomData<fn() -> T>,
}

impl<T: CanonicalEncode> CanonicalBuffer<T> {
    /// Allocate a reusable buffer of exactly `T::SIZE` bytes.
    pub fn new() -> Self {
        Self {
            bytes: vec![0; T::SIZE].into_boxed_slice(),
            marker: PhantomData,
        }
    }

    /// Encode `value` and return its canonical bytes borrowed from this buffer.
    ///
    /// # Panics
    ///
    /// Panics before returning bytes if the implementation writes fewer or more than `T::SIZE`
    /// bytes. The buffer remains reusable after catching such a panic, but no bytes from the failed
    /// encoding should be used.
    pub fn encode(&mut self, value: &T) -> &[u8] {
        let mut writer = CanonicalWriter::new(&mut self.bytes);
        value.encode(&mut writer);
        writer.finish::<T>();
        &self.bytes
    }
}

impl<T: CanonicalEncode> Default for CanonicalBuffer<T> {
    fn default() -> Self {
        Self::new()
    }
}

mod impls;

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::impls::test_support::*;

    struct SearchKey {
        generation: u64,
        selection: f64,
        counts: [u16; 3],
    }

    impl CanonicalEncode for SearchKey {
        const SIZE: usize = u64::SIZE + f64::SIZE + <[u16; 3]>::SIZE;

        fn encode(&self, writer: &mut CanonicalWriter<'_>) {
            writer
                .write(&self.generation)
                .write(&self.selection)
                .write(&self.counts);
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

    struct TooShort;

    impl CanonicalEncode for TooShort {
        const SIZE: usize = 2;

        fn encode(&self, writer: &mut CanonicalWriter<'_>) {
            writer.write_bytes(&[0x01]);
        }
    }

    struct TooLong;

    impl CanonicalEncode for TooLong {
        const SIZE: usize = 2;

        fn encode(&self, writer: &mut CanonicalWriter<'_>) {
            writer.write_bytes(&[0x01, 0x02, 0x03]);
        }
    }

    struct NestedTooShort;

    impl CanonicalEncode for NestedTooShort {
        const SIZE: usize = TooShort::SIZE;

        fn encode(&self, writer: &mut CanonicalWriter<'_>) {
            writer.write(&TooShort);
        }
    }

    struct ParentTooSmall;

    impl CanonicalEncode for ParentTooSmall {
        const SIZE: usize = 1;

        fn encode(&self, writer: &mut CanonicalWriter<'_>) {
            writer.write(&0u16);
        }
    }

    struct MaximumWidth;

    impl CanonicalEncode for MaximumWidth {
        const SIZE: usize = usize::MAX;

        fn encode(&self, _: &mut CanonicalWriter<'_>) {}
    }

    struct NestedSizeOverflow;

    impl CanonicalEncode for NestedSizeOverflow {
        const SIZE: usize = 1;

        fn encode(&self, writer: &mut CanonicalWriter<'_>) {
            writer.write_bytes(&[0x01]).write(&MaximumWidth);
        }
    }

    #[test]
    #[should_panic(expected = "CanonicalEncode implementation for")]
    fn buffer_rejects_short_encoding() {
        CanonicalBuffer::<TooShort>::new().encode(&TooShort);
    }

    #[test]
    #[should_panic(expected = "canonical encoder wrote past its declared size")]
    fn buffer_rejects_long_encoding() {
        CanonicalBuffer::<TooLong>::new().encode(&TooLong);
    }

    #[test]
    #[should_panic(expected = "CanonicalEncode implementation for")]
    fn nested_field_is_validated_before_parent_continues() {
        CanonicalBuffer::<NestedTooShort>::new().encode(&NestedTooShort);
    }

    #[test]
    #[should_panic(expected = "canonical encoder wrote past its declared size")]
    fn nested_field_cannot_exceed_parent_width() {
        CanonicalBuffer::<ParentTooSmall>::new().encode(&ParentTooSmall);
    }

    #[test]
    #[should_panic(expected = "canonical encoding length overflow")]
    fn nested_field_size_cannot_overflow_position() {
        CanonicalBuffer::<NestedSizeOverflow>::new().encode(&NestedSizeOverflow);
    }

    struct SometimesShort(bool);

    impl CanonicalEncode for SometimesShort {
        const SIZE: usize = 2;

        fn encode(&self, writer: &mut CanonicalWriter<'_>) {
            if self.0 {
                writer.write_bytes(&[0x01, 0x02]);
            } else {
                writer.write_bytes(&[0x03]);
            }
        }
    }

    #[test]
    fn reused_buffer_never_returns_a_partial_key() {
        use std::panic::{AssertUnwindSafe, catch_unwind};

        let mut buffer = CanonicalBuffer::<SometimesShort>::default();
        assert_eq!(buffer.encode(&SometimesShort(true)), [0x01, 0x02]);

        let failed = catch_unwind(AssertUnwindSafe(|| {
            let _ = buffer.encode(&SometimesShort(false));
        }));
        assert!(failed.is_err());

        assert_eq!(buffer.encode(&SometimesShort(true)), [0x01, 0x02]);
    }
}
