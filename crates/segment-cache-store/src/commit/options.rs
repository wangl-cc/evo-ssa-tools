//! Per-commit write policy.

use std::num::NonZeroUsize;

#[cfg(feature = "value-compression")]
use crate::block::ValuePayloadCompressionPolicy;

const DEFAULT_PATCH_DIRECT_RECORD_LIMIT: usize = 4_096;

/// Options consumed by one batch commit.
///
/// These fields affect newly written segment files only. They are not part of
/// the namespace identity and are not required when opening an existing store.
#[derive(Clone, Debug)]
pub struct CommitOptions {
    target_block_size: u32,
    #[cfg(feature = "value-compression")]
    value_payload_compression_policy: ValuePayloadCompressionPolicy,
    flush_threshold_records: NonZeroUsize,
    flush_threshold_bytes: NonZeroUsize,
    patch_direct_record_limit: usize,
}

impl Default for CommitOptions {
    fn default() -> Self {
        Self {
            target_block_size: 16 * 1024,
            #[cfg(feature = "value-compression")]
            value_payload_compression_policy: ValuePayloadCompressionPolicy::DEFAULT,
            flush_threshold_records: NonZeroUsize::new(4_096)
                .expect("default record threshold is non-zero"),
            flush_threshold_bytes: NonZeroUsize::new(8 * 1024 * 1024)
                .expect("default byte threshold is non-zero"),
            patch_direct_record_limit: DEFAULT_PATCH_DIRECT_RECORD_LIMIT,
        }
    }
}

impl CommitOptions {
    /// Sets the target logical block split size for newly written segments.
    pub fn with_target_block_size(mut self, target_block_size: u32) -> Self {
        self.target_block_size = target_block_size;
        self
    }

    /// Sets the writer-side policy for deciding whether value payloads are worth compressing.
    ///
    /// The store's persisted compression kind still controls which frame
    /// encodings are supported. This policy only affects newly written blocks
    /// whose store was created with a compression-capable kind.
    #[cfg(feature = "value-compression")]
    pub fn with_value_payload_compression_policy(
        mut self,
        policy: ValuePayloadCompressionPolicy,
    ) -> Self {
        self.value_payload_compression_policy = policy;
        self
    }

    /// Sets the maximum records written to one segment chunk during this commit.
    pub fn with_flush_threshold_records(mut self, flush_threshold_records: NonZeroUsize) -> Self {
        self.flush_threshold_records = flush_threshold_records;
        self
    }

    /// Sets the approximate maximum key/value bytes written to one segment chunk.
    pub fn with_flush_threshold_bytes(mut self, flush_threshold_bytes: NonZeroUsize) -> Self {
        self.flush_threshold_bytes = flush_threshold_bytes;
        self
    }

    /// Sets the maximum input records eligible for direct patch publication.
    ///
    /// A value of `0` disables direct patch publication for overlapping writes:
    /// every overlapping commit normalizes immediately.
    pub fn with_patch_direct_record_limit(mut self, patch_direct_record_limit: usize) -> Self {
        self.patch_direct_record_limit = patch_direct_record_limit;
        self
    }

    pub(super) fn target_block_size(&self) -> usize {
        self.target_block_size as usize
    }

    #[cfg(feature = "value-compression")]
    pub(super) fn value_payload_compression_policy(&self) -> ValuePayloadCompressionPolicy {
        self.value_payload_compression_policy
    }

    pub(super) fn flush_threshold_records(&self) -> usize {
        self.flush_threshold_records.get()
    }

    pub(super) fn flush_threshold_bytes(&self) -> usize {
        self.flush_threshold_bytes.get()
    }

    pub(super) fn patch_direct_record_limit(&self) -> usize {
        self.patch_direct_record_limit
    }
}
