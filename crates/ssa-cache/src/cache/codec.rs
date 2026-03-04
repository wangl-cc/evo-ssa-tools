use crate::Result;

/// Scratch buffer used by a [`Codec`] implementation.
///
/// Buffers are intended to be reused across many encode/decode operations to reduce allocations
/// during batched execution.
pub trait CodecBuffer {
    /// Create a new empty buffer.
    fn init() -> Self;
}

/// Serialization/deserialization used for cached values.
///
/// Implementations are expected to support buffer reuse: `encode` writes into `buffer` and returns
/// a byte slice that is valid until the next operation that mutates the same buffer.
pub trait Codec: Sized {
    /// Scratch buffer type used by this codec.
    type Buffer: CodecBuffer;

    /// Encode `self` into `buffer` and return the encoded bytes.
    fn encode<'b>(&self, buffer: &'b mut Self::Buffer) -> &'b [u8];
    /// Decode an instance from `bytes`, potentially reusing `buffer`.
    fn decode(bytes: &[u8], buffer: &mut Self::Buffer) -> Result<Self>;
}

#[cfg(feature = "bitcode")]
pub mod bitcode_codec {
    use super::*;

    impl CodecBuffer for bitcode::Buffer {
        fn init() -> Self {
            Self::new()
        }
    }

    impl<T> Codec for T
    where
        T: bitcode::Encode + for<'b> bitcode::Decode<'b>,
    {
        type Buffer = bitcode::Buffer;

        fn encode<'b>(&self, buffer: &'b mut Self::Buffer) -> &'b [u8] {
            buffer.encode(self)
        }

        fn decode(bytes: &[u8], buffer: &mut Self::Buffer) -> Result<Self> {
            Ok(buffer.decode(bytes)?)
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::{Codec, CodecBuffer};
    use crate::Result;

    #[cfg(feature = "bitcode")]
    #[test]
    fn bitcode_codec() -> Result<()> {
        let mut encode_buffer = <u64 as Codec>::Buffer::init();
        let encoded = 1024u64.encode(&mut encode_buffer).to_vec();

        let mut decode_buffer = <u64 as Codec>::Buffer::init();
        let decoded: u64 = u64::decode(&encoded, &mut decode_buffer)?;
        assert_eq!(decoded, 1024);

        let decoded: Result<u8> = u8::decode(&encoded, &mut decode_buffer);
        assert!(matches!(decoded.unwrap_err(), crate::Error::BitCode(_)));

        Ok(())
    }
}
