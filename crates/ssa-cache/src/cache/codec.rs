use crate::Result;

/// An abstract trait for a codec engine
pub trait CodecEngine: Default {}

pub trait Encode<T>: CodecEngine {
    fn encode<'a>(&'a mut self, value: &T) -> &'a [u8];
}

pub trait Decode<T>: CodecEngine {
    fn decode(&mut self, bytes: &[u8]) -> Result<T>;
}

pub trait Codec<T>: Encode<T> + Decode<T> {}

impl<T, E> Codec<T> for E where E: Encode<T> + Decode<T> {}

#[cfg(feature = "bitcode")]
pub mod bitcode_codec {
    use super::*;

    #[derive(Default)]
    pub struct BitcodeCodec {
        buffer: bitcode::Buffer,
    }

    impl CodecEngine for BitcodeCodec {}

    impl<T> Encode<T> for BitcodeCodec
    where
        T: bitcode::Encode,
    {
        fn encode(&mut self, value: &T) -> &[u8] {
            self.buffer.encode(value)
        }
    }

    impl<T> Decode<T> for BitcodeCodec
    where
        T: for<'b> bitcode::Decode<'b>,
    {
        fn decode(&mut self, bytes: &[u8]) -> Result<T> {
            Ok(self.buffer.decode(bytes)?)
        }
    }
}

#[cfg(feature = "bitcode")]
pub type DefaultCodec = bitcode_codec::BitcodeCodec;

#[cfg(test)]
mod tests {
    use super::{Decode, Encode};
    use crate::Result;

    #[cfg(feature = "bitcode")]
    #[test]
    fn test_codec() -> Result<()> {
        use super::bitcode_codec::BitcodeCodec;

        let mut engine = BitcodeCodec::default();

        let encoded = engine.encode(&1024u64).to_vec();
        let decoded: u64 = engine.decode(&encoded)?;
        assert_eq!(decoded, 1024);

        let decoded: Result<u8> = engine.decode(&encoded);
        assert!(matches!(decoded.unwrap_err(), crate::Error::BitCode(_)));

        Ok(())
    }
}
