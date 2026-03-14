use super::super::{CodecEngine, Error as CodecError, SkipReason};

/// A codec engine that uses the `bitcode 0.6` wire format for encoding and decoding.
///
/// This type pins the built-in `bitcode` backend to the current `0.6` format generation.
///
/// Use [`Bitcode06`] when the on-disk format should stay on this built-in generation instead of
/// following whatever backend [`Bitcode`] points to in a future crate release.
#[derive(Default)]
pub struct Bitcode06 {
    buffer: bitcode::Buffer,
}

/// Alias for the latest built-in `bitcode` backend.
///
/// `Bitcode` tracks the newest `bitcode` backend exposed by this crate. It is provided for
/// convenience, not as a stable wire-format name.
///
/// Compatibility notes:
///
/// - No wire-format compatibility is guaranteed across `bitcode` crate upgrades.
/// - Not recommended for persistent storage intended to outlive application upgrades.
/// - Do not use this alias when the exact on-disk format matters.
/// - Prefer versioned names such as [`Bitcode06`] when you need to pin the built-in bitcode format
///   generation in code.
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
