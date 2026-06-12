//! In-memory little-endian decoder for on-disk byte layouts.

/// Cursor over a byte slice being decoded as little-endian persisted data.
pub(crate) struct BinaryCursor<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> BinaryCursor<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    pub(crate) fn at(bytes: &'a [u8], position: usize) -> Self {
        Self { bytes, position }
    }

    pub(crate) fn seek(&mut self, position: usize) {
        self.position = position;
    }

    #[inline]
    pub(crate) fn read<T: LittleEndianValue>(&mut self) -> Option<T> {
        // `first_chunk::<WIDTH>` carries the length in the return type, so the
        // value decode needs no second length check. This compiles to the same
        // single bounds-checked load as a raw pointer read without relying on the
        // optimizer to fold away a redundant `TryFrom` length check.
        let value = T::read_le_prefix(self.bytes.get(self.position..)?)?;
        self.position += T::WIDTH;
        Some(value)
    }

    #[inline]
    pub(crate) fn read_vec(&mut self, len: usize) -> Option<Vec<u8>> {
        Some(self.read_slice(len)?.to_vec())
    }

    #[inline]
    pub(crate) fn read_slice(&mut self, len: usize) -> Option<&'a [u8]> {
        let start = self.position;
        let end = start.checked_add(len)?;
        let value = self.bytes.get(start..end)?;
        self.position = end;
        Some(value)
    }

    pub(crate) fn position(&self) -> usize {
        self.position
    }

    pub(crate) fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.position)
    }
}

/// Fixed-width scalar that can be decoded from little-endian persisted bytes.
pub(crate) trait LittleEndianValue: Sized {
    const WIDTH: usize;

    /// Decodes a value from the first `Self::WIDTH` bytes of `bytes`.
    ///
    /// Returns `None` when `bytes` is shorter than `Self::WIDTH`.
    fn read_le_prefix(bytes: &[u8]) -> Option<Self>;
}

impl LittleEndianValue for u32 {
    const WIDTH: usize = 4;

    #[inline]
    fn read_le_prefix(bytes: &[u8]) -> Option<Self> {
        Some(Self::from_le_bytes(*bytes.first_chunk::<4>()?))
    }
}

impl LittleEndianValue for u64 {
    const WIDTH: usize = 8;

    #[inline]
    fn read_le_prefix(bytes: &[u8]) -> Option<Self> {
        Some(Self::from_le_bytes(*bytes.first_chunk::<8>()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod fixed_width_reads {
        use super::*;

        #[test]
        fn read_uint_rejects_short_inputs() {
            let mut cursor = BinaryCursor::new(&[]);
            assert_eq!(cursor.read::<u32>(), None);
            assert_eq!(cursor.position(), 0);

            let mut cursor = BinaryCursor::new(&[1, 2, 3]);
            assert_eq!(cursor.read::<u32>(), None);
            assert_eq!(cursor.position(), 0);

            let mut cursor = BinaryCursor::new(&[1, 2, 3, 4, 5]);
            assert_eq!(cursor.read::<u64>(), None);
            assert_eq!(cursor.position(), 0);
        }

        #[test]
        fn read_u32_advances_after_success() {
            let mut cursor = BinaryCursor::at(&[0, 1, 2, 3, 4], 1);
            assert_eq!(cursor.read::<u32>(), Some(u32::from_le_bytes([1, 2, 3, 4])));
            assert_eq!(cursor.position(), 5);
        }

        #[test]
        fn read_u64_advances_after_success() {
            let mut cursor = BinaryCursor::at(&[0, 1, 2, 3, 4, 5, 6, 7, 8], 1);
            assert_eq!(
                cursor.read::<u64>(),
                Some(u64::from_le_bytes([1, 2, 3, 4, 5, 6, 7, 8]))
            );
            assert_eq!(cursor.position(), 9);
        }
    }

    mod byte_reads {
        use super::*;

        #[test]
        fn read_slice_rejects_short_inputs() {
            let mut cursor = BinaryCursor::new(&[1, 2]);
            assert_eq!(cursor.read_slice(3), None);
            assert_eq!(cursor.position(), 0);
        }

        #[test]
        fn read_slice_allows_zero_len_at_end() {
            let mut cursor = BinaryCursor::at(&[1, 2], 2);
            assert_eq!(cursor.read_slice(0), Some([].as_slice()));
            assert_eq!(cursor.position(), 2);
        }
    }
}
