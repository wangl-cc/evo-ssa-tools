mod collection;
mod primitive;

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
pub(crate) mod test_support {
    use crate::{CanonicalBuffer, CanonicalEncode};

    pub(crate) fn canonical_encode_size<T: CanonicalEncode>(_: &T) -> usize {
        T::SIZE
    }

    pub(crate) fn encode<T: CanonicalEncode>(value: &T) -> Vec<u8> {
        let mut buffer = CanonicalBuffer::<T>::new();
        buffer.encode(value).to_vec()
    }

    /// Assert that encoding `values` in order produces strictly increasing byte sequences.
    #[track_caller]
    pub(crate) fn assert_order_preserving<T: CanonicalEncode + std::fmt::Debug>(values: &[T]) {
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

    macro_rules! assert_encode {
        ($value:expr, [$($byte:literal),* $(,)?]) => {{
            let value = $value;
            let expected: &[u8] = &[$($byte),*];
            assert_eq!(
                expected.len(),
                $crate::impls::test_support::canonical_encode_size(&value)
            );

            let mut buffer = $crate::CanonicalBuffer::new();
            let encoded = buffer.encode(&value);
            assert_eq!(encoded, expected);
        }};
    }

    pub(crate) use assert_encode;
}
