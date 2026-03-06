use super::{CodecEngine, SkipReason};
use crate::Result;

/// A codec engine that uses [`bitcode`] for encoding and decoding.
#[derive(Default)]
pub struct Bitcode {
    buffer: bitcode::Buffer,
}

impl<T> CodecEngine<T> for Bitcode
where
    T: bitcode::Encode + for<'b> bitcode::Decode<'b>,
{
    fn encode(&mut self, value: &T) -> Result<&[u8], SkipReason> {
        Ok(self.buffer.encode(value))
    }

    fn decode(&mut self, bytes: &[u8]) -> Result<T> {
        Ok(self.buffer.decode(bytes)?)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::Result;

    #[test]
    fn encode_decode_roundtrip() -> Result<()> {
        let mut engine = Bitcode::default();
        let encoded = engine.encode(&1024u64).unwrap().to_vec();
        let decoded: u64 = engine.decode(&encoded)?;
        assert_eq!(decoded, 1024);
        Ok(())
    }

    #[test]
    fn decode_wrong_type_returns_error() -> Result<()> {
        let mut engine = Bitcode::default();
        let encoded = engine.encode(&1024u64).unwrap().to_vec();
        let result: crate::error::Result<u8> = engine.decode(&encoded);
        assert!(matches!(result.unwrap_err(), crate::Error::BitCode(_)));
        Ok(())
    }
}
