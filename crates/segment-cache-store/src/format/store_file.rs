//! `STORE` descriptor encoding and validation.
//!
//! `STORE` is line-oriented text and is written once at creation; it is the
//! creation completion marker and owns persistent identity (metadata, key
//! length, value layout). Reading and atomically publishing the file are
//! engine concerns.

use crate::format::{
    CatalogError, CatalogMismatch, StoreMetadata, ValueLayout, metadata::MetadataParseError,
};

const STORE_VERSION: u32 = 1;

const STORE_MAGIC: &str = "segment-cache-store store v1";

/// Malformed `STORE` descriptor bytes.
#[derive(thiserror::Error, Debug, Eq, PartialEq)]
pub enum StoreFileParseError {
    #[error("malformed STORE file: empty file")]
    Empty,

    #[error("malformed STORE file: unsupported magic")]
    UnsupportedMagic,

    #[error("malformed STORE file: missing required field {field}")]
    MissingField { field: &'static str },

    #[error("malformed STORE file: malformed field {field}")]
    MalformedField { field: &'static str },

    #[error("malformed STORE file: expected field {expected}, got {actual}")]
    UnexpectedFieldOrder {
        expected: &'static str,
        actual: String,
    },

    #[error("malformed STORE file: invalid value for field {field}")]
    InvalidFieldValue { field: &'static str },

    #[error("malformed STORE file: unexpected trailing fields")]
    UnexpectedTrailingFields,

    #[error(transparent)]
    Metadata(#[from] MetadataParseError),
}

/// Persistent store identity stored in `STORE`.
#[derive(Clone, Debug)]
pub(crate) struct StoreDescriptor {
    pub(crate) version: u32,
    pub(crate) metadata: StoreMetadata,
    pub(crate) key_len: usize,
    pub(crate) value_layout: ValueLayout,
    pub(crate) block_checksum_id: u32,
    pub(crate) value_payload_compression_id: u32,
}

impl StoreDescriptor {
    pub(crate) fn new(
        metadata: StoreMetadata,
        key_len: usize,
        value_layout: ValueLayout,
        block_checksum_id: u32,
        value_payload_compression_id: u32,
    ) -> Self {
        Self {
            version: STORE_VERSION,
            metadata,
            key_len,
            value_layout,
            block_checksum_id,
            value_payload_compression_id,
        }
    }

    pub(crate) fn encode(&self) -> String {
        let mut out = String::new();
        push_line(&mut out, STORE_MAGIC);
        push_line(&mut out, &format!("version={}", self.version));
        push_line(
            &mut out,
            &format!("metadata={}", self.metadata.encode_store_value()),
        );
        push_line(&mut out, &format!("key_len={}", self.key_len));
        push_line(
            &mut out,
            &format!("value_len={}", self.value_layout.to_u32()),
        );
        push_line(
            &mut out,
            &format!("block_checksum_id={}", self.block_checksum_id),
        );
        push_line(
            &mut out,
            &format!(
                "value_payload_compression_id={}",
                self.value_payload_compression_id
            ),
        );
        out
    }

    pub(crate) fn parse(input: &str) -> std::result::Result<Self, StoreFileParseError> {
        StoreParser::new(input).parse()
    }

    pub(crate) fn validate_structure(&self) -> std::result::Result<(), CatalogError> {
        if self.version != STORE_VERSION {
            return Err(CatalogError::UnsupportedVersion {
                file: "STORE",
                version: self.version,
            });
        }
        if self.key_len == 0 {
            return Err(CatalogMismatch::StoreKeyLenZero.into());
        }
        if self.key_len > u32::MAX as usize {
            return Err(CatalogMismatch::StoreKeyLenTooLarge.into());
        }
        Ok(())
    }
}

struct StoreParser<'a> {
    lines: std::str::Lines<'a>,
}

impl<'a> StoreParser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            lines: input.lines(),
        }
    }

    fn parse(mut self) -> std::result::Result<StoreDescriptor, StoreFileParseError> {
        self.expect_magic()?;
        let version = self.required_value::<u32>("version")?;
        let metadata = StoreMetadata::parse_store_value(self.required_str("metadata")?)?;
        let key_len = self.required_value::<usize>("key_len")?;
        let value_len = self.required_value::<u32>("value_len")?;
        let block_checksum_id = self.required_value::<u32>("block_checksum_id")?;
        let value_payload_compression_id =
            self.required_value::<u32>("value_payload_compression_id")?;
        if self.lines.any(|line| !line.is_empty()) {
            return Err(StoreFileParseError::UnexpectedTrailingFields);
        }
        Ok(StoreDescriptor {
            version,
            metadata,
            key_len,
            value_layout: ValueLayout::from_u32(value_len),
            block_checksum_id,
            value_payload_compression_id,
        })
    }

    fn expect_magic(&mut self) -> std::result::Result<(), StoreFileParseError> {
        let Some(magic) = self.lines.next() else {
            return Err(StoreFileParseError::Empty);
        };
        if magic != STORE_MAGIC {
            return Err(StoreFileParseError::UnsupportedMagic);
        }
        Ok(())
    }

    fn required_str(
        &mut self,
        key: &'static str,
    ) -> std::result::Result<&'a str, StoreFileParseError> {
        let Some(line) = self.lines.next() else {
            return Err(StoreFileParseError::MissingField { field: key });
        };
        let Some((actual_key, value)) = line.split_once('=') else {
            return Err(StoreFileParseError::MalformedField { field: key });
        };
        if actual_key != key {
            return Err(StoreFileParseError::UnexpectedFieldOrder {
                expected: key,
                actual: actual_key.to_owned(),
            });
        }
        Ok(value)
    }

    fn required_value<T>(
        &mut self,
        key: &'static str,
    ) -> std::result::Result<T, StoreFileParseError>
    where
        T: std::str::FromStr,
    {
        self.required_str(key)?
            .parse()
            .map_err(|_| StoreFileParseError::InvalidFieldValue { field: key })
    }
}

fn push_line(out: &mut String, line: &str) {
    out.push_str(line);
    out.push('\n');
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::ValuePayloadCompressionKind;

    mod store_file {
        use super::*;

        fn store_file_text() -> String {
            StoreDescriptor::new(
                StoreMetadata::from_text("meta"),
                16,
                ValueLayout::VARIABLE,
                1,
                ValuePayloadCompressionKind::None.format_id(),
            )
            .encode()
        }

        #[test]
        fn round_trips_store_descriptor() {
            let descriptor = StoreDescriptor::new(
                StoreMetadata::from_text("meta"),
                16,
                ValueLayout::fixed(std::num::NonZeroU32::new(32).expect("non-zero")),
                1,
                ValuePayloadCompressionKind::None.format_id(),
            );
            let parsed = StoreDescriptor::parse(&descriptor.encode()).expect("store should parse");
            assert_eq!(parsed.metadata, descriptor.metadata);
            assert_eq!(parsed.key_len, 16);
            assert_eq!(parsed.block_checksum_id, 1);
            assert_eq!(
                parsed.value_payload_compression_id,
                ValuePayloadCompressionKind::None.format_id()
            );
            assert_eq!(
                parsed.value_layout,
                ValueLayout::fixed(std::num::NonZeroU32::new(32).expect("non-zero"))
            );
        }

        #[test]
        fn rejects_malformed_store_file() {
            assert!(matches!(
                StoreDescriptor::parse("bad\n"),
                Err(StoreFileParseError::UnsupportedMagic)
            ));
        }

        #[test]
        fn rejects_store_file_field_shape_errors() {
            assert!(matches!(
                StoreDescriptor::parse(""),
                Err(StoreFileParseError::Empty)
            ));
            assert!(matches!(
                StoreDescriptor::parse("segment-cache-store store v1\n"),
                Err(StoreFileParseError::MissingField { field: "version" })
            ));

            let malformed = store_file_text().replace("version=1\n", "version\n");
            assert!(matches!(
                StoreDescriptor::parse(&malformed),
                Err(StoreFileParseError::MalformedField { field: "version" })
            ));

            let wrong_order =
                store_file_text().replace("metadata=6d657461\nkey_len=16\n", "key_len=16\n");
            assert!(matches!(
                StoreDescriptor::parse(&wrong_order),
                Err(StoreFileParseError::UnexpectedFieldOrder {
                    expected: "metadata",
                    ..
                })
            ));

            let invalid_value = store_file_text().replace("key_len=16\n", "key_len=not-a-len\n");
            assert!(matches!(
                StoreDescriptor::parse(&invalid_value),
                Err(StoreFileParseError::InvalidFieldValue { field: "key_len" })
            ));

            let mut trailing = store_file_text();
            trailing.push_str("extra=field\n");
            assert!(matches!(
                StoreDescriptor::parse(&trailing),
                Err(StoreFileParseError::UnexpectedTrailingFields)
            ));
        }

        #[test]
        fn rejects_invalid_store_descriptor_structure() {
            let mut descriptor = StoreDescriptor::new(
                StoreMetadata::from_text("meta"),
                16,
                ValueLayout::VARIABLE,
                1,
                ValuePayloadCompressionKind::None.format_id(),
            );

            descriptor.version = 2;
            assert!(matches!(
                descriptor.validate_structure(),
                Err(CatalogError::UnsupportedVersion {
                    file: "STORE",
                    version: 2
                })
            ));

            descriptor.version = 1;
            descriptor.key_len = 0;
            assert!(matches!(
                descriptor.validate_structure(),
                Err(CatalogError::Mismatch(CatalogMismatch::StoreKeyLenZero))
            ));

            descriptor.key_len = u32::MAX as usize + 1;
            assert!(matches!(
                descriptor.validate_structure(),
                Err(CatalogError::Mismatch(CatalogMismatch::StoreKeyLenTooLarge))
            ));
        }
    }
}
