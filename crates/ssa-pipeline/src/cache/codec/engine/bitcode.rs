use super::super::{CodecEngine, Error as CodecError, SkipReason};

/// A codec engine using `bitcode v0.6` for serialization and deserialization.
///
/// `bitcode` is a fast and compact binary format for serializing Rust data structures, but it does
/// not guarantee compatibility across major versions. This codec is therefore named `Bitcode06`
/// explicitly.
///
/// Migrating stored data to a different `bitcode` version should be treated as a data migration.
#[derive(Default)]
pub struct Bitcode06 {
    buffer: bitcode::Buffer,
}

/// Alias for the latest `bitcode` codec implementation.
///
/// `Bitcode` always refers to the latest `bitcode` codec exposed by this crate. It is provided for
/// convenience and no wire-format compatibility is guaranteed across `bitcode` crate upgrades.
///
/// This alias must **NOT** be use this alias for **persistent storage**. Use this only for
/// in-memory or other volatile caches where data can be discarded freely. Prefer versioned names
/// such as [`Bitcode06`] everywhere else.
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
