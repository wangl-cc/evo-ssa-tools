//! Store configuration and value-layout compatibility helpers.

use std::path::PathBuf;

use crate::error::{Error, Result};

/// Physical value layout used by every segment in one store.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValueLayout {
    /// Values are opaque byte slices with per-record lengths.
    Variable,
    /// Every value must have exactly `value_len` bytes.
    Fixed { value_len: usize },
}

/// Configuration for opening or creating a store.
#[derive(Clone, Debug)]
pub struct StoreOptions {
    /// Store root directory.
    pub root: PathBuf,
    /// Fixed key length in bytes.
    pub key_len: usize,
    /// Value layout shared by all visible segments.
    pub value_layout: ValueLayout,
    /// Fixed shard count for this store.
    pub shard_count: usize,
    /// Byte offset where lexicographic-prefix sharding starts.
    pub shard_key_offset: usize,
    /// Store-level codec version; mismatched segments are ignored.
    pub codec_version: u32,
    /// Whether data-block CRC32C checksums are verified on read.
    pub verify_block_checksums: bool,
    /// Target physical block size for newly written segments.
    pub target_block_size: usize,
    /// Maximum records per newly published segment chunk.
    pub flush_threshold_records: usize,
    /// Maximum approximate key/value bytes per newly published segment chunk.
    pub flush_threshold_bytes: usize,
}

impl StoreOptions {
    /// Creates default options for a store rooted at `root` with fixed-width keys.
    #[must_use]
    pub fn new(root: impl Into<PathBuf>, key_len: usize) -> Self {
        Self {
            root: root.into(),
            key_len,
            value_layout: ValueLayout::Variable,
            shard_count: 8,
            shard_key_offset: 16,
            codec_version: 0,
            verify_block_checksums: true,
            target_block_size: 16 * 1024,
            flush_threshold_records: 4_096,
            flush_threshold_bytes: 8 * 1024 * 1024,
        }
    }

    /// Selects the physical value layout used by all visible segments.
    #[must_use]
    pub fn with_value_layout(mut self, value_layout: ValueLayout) -> Self {
        self.value_layout = value_layout;
        self
    }

    /// Enables fixed-value layout and rejects future writes with any other value length.
    #[must_use]
    pub fn with_fixed_value_len(mut self, value_len: usize) -> Self {
        self.value_layout = ValueLayout::Fixed { value_len };
        self
    }

    /// Sets the fixed shard count persisted in the manifest.
    #[must_use]
    pub fn with_shard_count(mut self, shard_count: usize) -> Self {
        self.shard_count = shard_count;
        self
    }

    /// Sets the byte offset where shard assignment starts reading key bytes.
    #[must_use]
    pub fn with_shard_key_offset(mut self, shard_key_offset: usize) -> Self {
        self.shard_key_offset = shard_key_offset;
        self
    }

    /// Sets the store-level codec version stored in each segment footer.
    #[must_use]
    pub fn with_codec_version(mut self, codec_version: u32) -> Self {
        self.codec_version = codec_version;
        self
    }

    /// Enables or disables block checksum verification on reads.
    #[must_use]
    pub fn with_block_checksum_verification(mut self, verify: bool) -> Self {
        self.verify_block_checksums = verify;
        self
    }

    /// Sets the target physical block size for newly written segments.
    #[must_use]
    pub fn with_target_block_size(mut self, target_block_size: usize) -> Self {
        self.target_block_size = target_block_size;
        self
    }

    /// Sets the maximum records written to one segment chunk during a batch commit.
    #[must_use]
    pub fn with_flush_threshold_records(mut self, flush_threshold_records: usize) -> Self {
        self.flush_threshold_records = flush_threshold_records;
        self
    }

    /// Sets the approximate maximum key/value bytes written to one segment chunk.
    #[must_use]
    pub fn with_flush_threshold_bytes(mut self, flush_threshold_bytes: usize) -> Self {
        self.flush_threshold_bytes = flush_threshold_bytes;
        self
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if self.key_len == 0 {
            return Err(Error::InvalidOptions {
                reason: "key_len must be greater than zero",
            });
        }
        if self.shard_count == 0 {
            return Err(Error::InvalidOptions {
                reason: "shard_count must be greater than zero",
            });
        }
        if self.shard_count > usize::try_from(u32::MAX).expect("u32 max should fit in usize") {
            return Err(Error::InvalidOptions {
                reason: "shard_count must fit in u32",
            });
        }
        if self.key_len > usize::try_from(u32::MAX).expect("u32 max should fit in usize") {
            return Err(Error::InvalidOptions {
                reason: "key_len must fit in u32",
            });
        }
        if let ValueLayout::Fixed { value_len } = self.value_layout {
            if value_len == 0 {
                return Err(Error::InvalidOptions {
                    reason: "fixed value_len must be greater than zero",
                });
            }
            if value_len > usize::try_from(u32::MAX).expect("u32 max should fit in usize") {
                return Err(Error::InvalidOptions {
                    reason: "fixed value_len must fit in u32",
                });
            }
        }
        if self.shard_key_offset > self.key_len {
            return Err(Error::InvalidOptions {
                reason: "shard_key_offset must not exceed key_len",
            });
        }
        if self.flush_threshold_records == 0 {
            return Err(Error::InvalidOptions {
                reason: "flush_threshold_records must be greater than zero",
            });
        }
        if self.flush_threshold_bytes == 0 {
            return Err(Error::InvalidOptions {
                reason: "flush_threshold_bytes must be greater than zero",
            });
        }
        Ok(())
    }
}

impl ValueLayout {
    /// Encodes this layout into the line-oriented manifest value.
    pub(crate) fn encode_manifest_value(self) -> String {
        match self {
            Self::Variable => "variable".to_owned(),
            Self::Fixed { value_len } => format!("fixed:{value_len}"),
        }
    }

    /// Parses a manifest value-layout field.
    pub(crate) fn parse_manifest_value(value: &str) -> Result<Self> {
        if value == "variable" {
            return Ok(Self::Variable);
        }
        if let Some(value_len) = value.strip_prefix("fixed:") {
            let value_len = value_len.parse().map_err(|_| Error::ManifestParse {
                reason: "invalid fixed value length".to_owned(),
            })?;
            return Ok(Self::Fixed { value_len });
        }
        Err(Error::ManifestParse {
            reason: "invalid value layout".to_owned(),
        })
    }

    /// Numeric tag used by segment footers.
    pub(crate) fn segment_tag(self) -> u32 {
        match self {
            Self::Variable => 0,
            Self::Fixed { .. } => 1,
        }
    }

    /// Fixed value length field used by segment footers.
    pub(crate) fn segment_fixed_len(self) -> u32 {
        match self {
            Self::Variable => 0,
            Self::Fixed { value_len } => {
                u32::try_from(value_len).expect("fixed value length should fit in u32")
            }
        }
    }

    /// Decodes the value-layout fields stored in a segment footer.
    pub(crate) fn from_segment_fields(tag: u32, fixed_len: u32) -> Result<Self> {
        match (tag, fixed_len) {
            (0, 0) => Ok(Self::Variable),
            (1, len) if len > 0 => Ok(Self::Fixed {
                value_len: usize::try_from(len).expect("fixed value length should fit"),
            }),
            _ => Err(Error::UnsupportedFormatVersion { version: 0 }),
        }
    }
}

#[cfg(test)]
mod tests {
    mod value_layout_metadata {
        use super::super::ValueLayout;
        use crate::Error;

        #[test]
        fn rejects_invalid_encodings() {
            assert_eq!(
                ValueLayout::parse_manifest_value("fixed:32").expect("fixed layout should parse"),
                ValueLayout::Fixed { value_len: 32 }
            );
            assert!(matches!(
                ValueLayout::parse_manifest_value("fixed:not-a-number"),
                Err(Error::ManifestParse { .. })
            ));
            assert!(matches!(
                ValueLayout::parse_manifest_value("unknown"),
                Err(Error::ManifestParse { .. })
            ));
            assert!(matches!(
                ValueLayout::from_segment_fields(7, 0),
                Err(Error::UnsupportedFormatVersion { .. })
            ));
        }
    }
}
