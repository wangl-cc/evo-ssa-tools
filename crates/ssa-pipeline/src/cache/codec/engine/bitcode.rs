use super::super::{CodecEngine, Error as CodecError, SkipReason};

/// A codec engine that uses the `bitcode v0.6` for encoding and decoding.
///
/// Prefer [`Bitcode06`] by default to avoid compatibility issues. If you later move stored data to
/// another `bitcode` version, treat that as an explicit migration step.
#[derive(Default)]
pub struct Bitcode06 {
    buffer: bitcode::Buffer,
}

/// Alias for the latest `bitcode` backend.
///
/// `Bitcode` tracks the newest `bitcode` backend exposed by this crate. It is provided for
/// convenience, no wire-format compatibility is guaranteed across `bitcode` crate upgrades.
/// **DO NOT** use this alias for **any persistent storage**. Use this only for in-memory or other
/// volatile caches where data can be discarded freely. Prefer versioned names such as [`Bitcode06`]
/// everywhere else.
pub type Bitcode = Bitcode06;

impl<T> CodecEngine<T> for Bitcode06
where
    T: bitcode::Encode + for<'b> bitcode::Decode<'b>,
{
    fn encode(&mut self, value: &T) -> Result<&[u8], SkipReason> {
        Ok(self.buffer.encode(value))
    }

    fn decode(&mut self, bytes: &[u8]) -> Result<T, CodecError> {
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
        let mut engine = Bitcode06::default();
        let encoded = engine.encode(&1024u64).unwrap().to_vec();
        let decoded: u64 = engine.decode(&encoded)?;
        assert_eq!(decoded, 1024);
        Ok(())
    }

    #[test]
    fn decode_wrong_type_returns_error() -> Result<()> {
        let mut engine = Bitcode06::default();
        let encoded = engine.encode(&1024u64).unwrap().to_vec();
        let result: Result<u8, _> = engine.decode(&encoded);
        assert!(matches!(result.unwrap_err(), CodecError::BitCode(_)));
        Ok(())
    }

    #[test]
    fn alias_matches_versioned_engine() -> Result<()> {
        let mut engine = Bitcode::default();
        let encoded = engine.encode(&1024u64).unwrap().to_vec();
        let decoded: u64 = engine.decode(&encoded)?;
        assert_eq!(decoded, 1024);
        Ok(())
    }
}
