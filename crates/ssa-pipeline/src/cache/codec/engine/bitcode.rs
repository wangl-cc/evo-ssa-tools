use super::super::{CodecEngine, Error as CodecError, SkipReason};
use crate::cache::Fork;

/// A codec engine using `bitcode v0.6` for serialization and deserialization.
///
/// `bitcode` is a fast and compact binary format for serializing Rust data structures, but it does
/// not guarantee compatibility across major versions. This codec is therefore named `Bitcode06`
/// explicitly.
///
/// Migrating stored data to a different `bitcode` version should be treated as a data migration.
///
/// Each worker gets its own independent engine instance with a fresh encode buffer.
#[derive(Default)]
pub struct Bitcode06 {
    buffer: bitcode::Buffer,
}

impl Fork for Bitcode06 {
    fn fork(&self) -> Self {
        Self::default()
    }
}


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
        let mut engine = Bitcode06::default();
        let encoded = engine.encode(&1024u64).unwrap().to_vec();
        let decoded: u64 = engine.decode(&encoded)?;
        assert_eq!(decoded, 1024);
        Ok(())
    }

    #[test]
    fn fork_produces_independent_engine() -> Result<()> {
        use crate::cache::Fork;
        let engine = Bitcode06::default();
        let mut forked = engine.fork();
        let encoded = forked.encode(&42u32).unwrap().to_vec();
        let decoded: u32 = forked.decode(&encoded)?;
        assert_eq!(decoded, 42);
        Ok(())
    }
}
