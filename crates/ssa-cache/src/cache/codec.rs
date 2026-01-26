use crate::Result;

pub trait CodecBuffer {
    fn init() -> Self;
}

pub trait Codec: Sized {
    type Buffer: CodecBuffer;

    fn encode<'b>(&self, buffer: &'b mut Self::Buffer) -> &'b [u8];
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

#[cfg(feature = "bitcode")]
pub type BitcodeCodec = bitcode::Buffer;

#[cfg(feature = "bitcode")]
pub type DefaultCodec = bitcode::Buffer;

#[cfg(test)]
mod tests {
    use super::{Codec, CodecBuffer};
    use crate::Result;

    #[cfg(feature = "bitcode")]
    #[test]
    fn test_codec() -> Result<()> {
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
