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

    #[must_use]
    pub fn with_value_layout(mut self, value_layout: ValueLayout) -> Self {
        self.value_layout = value_layout;
        self
    }

    #[must_use]
    pub fn with_fixed_value_len(mut self, value_len: usize) -> Self {
        self.value_layout = ValueLayout::Fixed { value_len };
        self
    }

    #[must_use]
    pub fn with_shard_count(mut self, shard_count: usize) -> Self {
        self.shard_count = shard_count;
        self
    }

    #[must_use]
    pub fn with_shard_key_offset(mut self, shard_key_offset: usize) -> Self {
        self.shard_key_offset = shard_key_offset;
        self
    }

    #[must_use]
    pub fn with_codec_version(mut self, codec_version: u32) -> Self {
        self.codec_version = codec_version;
        self
    }

    #[must_use]
    pub fn with_block_checksum_verification(mut self, verify: bool) -> Self {
        self.verify_block_checksums = verify;
        self
    }

    #[must_use]
    pub fn with_target_block_size(mut self, target_block_size: usize) -> Self {
        self.target_block_size = target_block_size;
        self
    }

    #[must_use]
    pub fn with_flush_threshold_records(mut self, flush_threshold_records: usize) -> Self {
        self.flush_threshold_records = flush_threshold_records;
        self
    }

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
