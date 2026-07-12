//! Opaque caller-defined store metadata and its `STORE` hex encoding.

/// Malformed `metadata=<hex>` value in `STORE`.
#[derive(thiserror::Error, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum MetadataParseError {
    #[error("metadata hex value has odd length")]
    OddLengthHex,

    #[error("metadata hex value contains an invalid digit")]
    InvalidHexDigit,
}

/// Opaque caller-defined compatibility metadata for one store root.
///
/// The storage layer persists these bytes in `STORE` and compares them when
/// requested at open time. Callers can put a namespace name, schema
/// fingerprint, codec description, or any other cache compatibility identifier
/// here without coupling it to the segment format.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct StoreMetadata {
    bytes: Vec<u8>,
}

impl StoreMetadata {
    /// Creates metadata from opaque bytes.
    pub fn from_bytes(bytes: impl Into<Vec<u8>>) -> Self {
        Self {
            bytes: bytes.into(),
        }
    }

    /// Creates metadata from a UTF-8 string.
    pub fn from_text(text: impl AsRef<str>) -> Self {
        Self::from_bytes(text.as_ref().as_bytes())
    }

    /// Returns the persisted metadata bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub(crate) fn encode_store_value(&self) -> String {
        HexBytes::encode(&self.bytes)
    }

    pub(crate) fn parse_store_value(value: &str) -> std::result::Result<Self, MetadataParseError> {
        Ok(Self::from_bytes(HexBytes::decode(value)?))
    }
}

struct HexBytes;

impl HexBytes {
    fn encode(bytes: &[u8]) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut out = String::with_capacity(bytes.len() * 2);
        for byte in bytes {
            out.push(char::from(HEX[usize::from(byte >> 4)]));
            out.push(char::from(HEX[usize::from(byte & 0x0f)]));
        }
        out
    }

    fn decode(value: &str) -> std::result::Result<Vec<u8>, MetadataParseError> {
        if !value.len().is_multiple_of(2) {
            return Err(MetadataParseError::OddLengthHex);
        }
        let mut bytes = Vec::with_capacity(value.len() / 2);
        for pair in value.as_bytes().chunks_exact(2) {
            bytes.push((Self::decode_nibble(pair[0])? << 4) | Self::decode_nibble(pair[1])?);
        }
        Ok(bytes)
    }

    fn decode_nibble(value: u8) -> std::result::Result<u8, MetadataParseError> {
        match value {
            b'0'..=b'9' => Ok(value - b'0'),
            b'a'..=b'f' => Ok(value - b'a' + 10),
            b'A'..=b'F' => Ok(value - b'A' + 10),
            _ => Err(MetadataParseError::InvalidHexDigit),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod hex_metadata {
        use super::*;

        #[test]
        fn round_trips_text_and_bytes() {
            let metadata = StoreMetadata::from_text("schema-v1");
            let encoded = metadata.encode_store_value();
            let parsed = StoreMetadata::parse_store_value(&encoded).expect("metadata parses");

            assert_eq!(parsed, metadata);
            assert_eq!(parsed.as_bytes(), b"schema-v1");
        }

        #[test]
        fn rejects_malformed_hex() {
            assert!(StoreMetadata::parse_store_value("abc").is_err());
            assert!(StoreMetadata::parse_store_value("xx").is_err());
        }
    }
}
