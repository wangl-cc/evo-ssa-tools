use serde::{Serialize, de::DeserializeOwned};

use super::super::{CodecEngine, Error as CodecError, SkipReason};

/// A codec engine using `postcard` for serialization and deserialization.
///
/// `postcard` is the built-in serde-based backend. It is a good fit when you want a stable codec
/// name in the API without tying the implementation to a versioned `bitcode` backend.
#[derive(Default)]
pub struct Postcard {
    buffer: Vec<u8>,
}

struct ReuseVecFlavor<'a> {
    buffer: &'a mut Vec<u8>,
}

impl<'a> ReuseVecFlavor<'a> {
    fn new(buffer: &'a mut Vec<u8>) -> Self {
        Self { buffer }
    }
}

impl postcard::ser_flavors::Flavor for ReuseVecFlavor<'_> {
    type Output = usize;

    #[inline(always)]
    fn try_push(&mut self, data: u8) -> postcard::Result<()> {
        self.buffer.push(data);
        Ok(())
    }

    #[inline(always)]
    fn try_extend(&mut self, data: &[u8]) -> postcard::Result<()> {
        self.buffer.extend_from_slice(data);
        Ok(())
    }

    #[inline(always)]
    fn finalize(self) -> postcard::Result<Self::Output> {
        Ok(self.buffer.len())
    }
}

impl<T> CodecEngine<T> for Postcard
where
    T: Serialize + DeserializeOwned,
{
    fn encode(&mut self, value: &T) -> Result<&[u8], SkipReason> {
        self.buffer.clear();
        let used = postcard::serialize_with_flavor(value, ReuseVecFlavor::new(&mut self.buffer))
            .map_err(|err| SkipReason::EncodeFailure(Box::new(err)))?;
        Ok(&self.buffer[..used])
    }

    fn decode(&mut self, bytes: &[u8]) -> Result<T, CodecError> {
        Ok(postcard::from_bytes(bytes)?)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use postcard::ser_flavors::Flavor;
    use serde::{
        Deserialize, Deserializer,
        ser::{Error as _, Serializer},
    };

    use super::*;
    use crate::Result;

    struct EncodeFails;

    impl Serialize for EncodeFails {
        fn serialize<S>(&self, _: S) -> core::result::Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            Err(S::Error::custom("expected test failure"))
        }
    }

    impl<'de> Deserialize<'de> for EncodeFails {
        fn deserialize<D>(deserializer: D) -> core::result::Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            let _ = u8::deserialize(deserializer)?;
            Ok(Self)
        }
    }

    #[test]
    fn encode_decode_roundtrip() -> Result<()> {
        let mut engine = Postcard::default();
        let encoded = engine.encode(&1024u64).unwrap().to_vec();
        let decoded: u64 = engine.decode(&encoded)?;
        assert_eq!(decoded, 1024);
        Ok(())
    }

    #[test]
    fn decode_wrong_type_returns_error() -> Result<()> {
        let mut engine = Postcard::default();
        let encoded = engine.encode(&[0xffu8]).unwrap().to_vec();
        let result: Result<char, _> = engine.decode(&encoded);
        assert!(matches!(result.unwrap_err(), CodecError::Postcard(_)));
        Ok(())
    }

    #[test]
    fn encode_failure_preserves_source_error() {
        let mut engine = Postcard::default();
        let err = engine.encode(&EncodeFails).unwrap_err();
        match err {
            SkipReason::EncodeFailure(source) => {
                assert_eq!(source.to_string(), "Serde Serialization Error");
            }
            other => panic!("expected EncodeFailure, got {other:?}"),
        }
    }

    #[test]
    fn reuse_vec_flavor_appends_and_reports_used_len() {
        let mut buffer = Vec::new();
        let mut flavor = ReuseVecFlavor::new(&mut buffer);
        flavor.try_push(1).unwrap();
        flavor.try_extend(&[2, 3]).unwrap();
        let used = flavor.finalize().unwrap();
        assert_eq!(used, 3);
        assert_eq!(buffer, [1, 2, 3]);
    }
}
