//! Open immutable segment state and process-local block verification cache.

use std::{cmp::Ordering, fs::File};

use parking_lot::Mutex;

use super::{
    SegmentGeometry,
    file::{BlockReadOptions, OpenedSegment, read_block, read_block_reusing},
    index::SegmentIndex,
    writer::SegmentFileMetadata,
};
use crate::{
    block::DecodedBlock,
    error::Result,
    key::{BlockRelativeKey, SegmentRelativeKey},
};

/// Visible immutable segment and its in-memory sparse block index.
pub(crate) struct Segment {
    segment_id: u32,
    file: File,
    min_key: Vec<u8>,
    max_key: Vec<u8>,
    block_index: SegmentIndex,
    block_verifications: Mutex<BlockVerificationCache>,
}

/// Process-local verification state for immutable segment blocks.
///
/// Verification advances monotonically from lookup metadata to payload bytes.
/// Later reads through the same open segment can skip completed phases. Segment
/// files are immutable while open, so dropping the [`Segment`] is the only
/// invalidation required.
#[derive(Debug)]
struct BlockVerificationCache {
    phases: Vec<BlockVerification>,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum BlockVerification {
    Unverified,
    LookupVerified,
    PayloadVerified,
}

impl Segment {
    pub(crate) fn id(&self) -> u32 {
        self.segment_id
    }

    pub(crate) fn min_key(&self) -> &[u8] {
        &self.min_key
    }

    pub(crate) fn max_key(&self) -> &[u8] {
        &self.max_key
    }

    pub(crate) fn block_count(&self) -> usize {
        self.block_index.len()
    }

    pub(crate) fn segment_prefix(&self) -> &[u8] {
        self.block_index.segment_prefix().as_slice()
    }

    /// Converts a key whose segment range membership was already established.
    pub(crate) fn relative_key_in_range<'a>(&self, key: &'a [u8]) -> SegmentRelativeKey<'a> {
        debug_assert!(self.min_key() <= key && key <= self.max_key());
        self.block_index.relative_key(key)
    }

    pub(crate) fn block_contains(&self, index: usize, key: SegmentRelativeKey<'_>) -> bool {
        self.block_index.contains(index, key)
    }

    pub(crate) fn block_min_cmp(&self, index: usize, key: SegmentRelativeKey<'_>) -> Ordering {
        self.block_index.relative_min_cmp(index, key)
    }

    pub(crate) fn block_max_cmp(&self, index: usize, key: SegmentRelativeKey<'_>) -> Ordering {
        self.block_index.relative_max_cmp(index, key)
    }

    pub(crate) fn block_relative_key<'a>(
        &self,
        index: usize,
        key: SegmentRelativeKey<'a>,
    ) -> BlockRelativeKey<'a> {
        self.block_index.block_relative_key(index, key)
    }

    /// Converts a verified on-disk segment into the state used by readers.
    pub(super) fn from_opened(segment_id: u32, opened: OpenedSegment) -> Self {
        let block_verifications = BlockVerificationCache::new(opened.block_index.len());
        Self {
            segment_id,
            file: opened.file,
            min_key: opened.min_key,
            max_key: opened.max_key,
            block_index: opened.block_index,
            block_verifications: Mutex::new(block_verifications),
        }
    }

    /// Builds state directly from newly written segment metadata.
    pub(crate) fn from_written(segment_id: u32, file: File, metadata: SegmentFileMetadata) -> Self {
        let min_key = metadata.min_key;
        let max_key = metadata.max_key;
        let footer = metadata.footer;
        let block_index = footer.block_index;
        let block_verifications = BlockVerificationCache::new(block_index.len());
        Self {
            segment_id,
            file,
            min_key,
            max_key,
            block_index,
            block_verifications: Mutex::new(block_verifications),
        }
    }

    /// Finds the sparse block-index entry that may contain `key`.
    pub(crate) fn find_block_index(&self, key: SegmentRelativeKey<'_>) -> usize {
        self.block_index.find_block(key)
    }

    /// Reads and decodes a block by sparse block-index position.
    pub(crate) fn load_block(
        &self,
        block_index: usize,
        geometry: SegmentGeometry,
        verify_checksum: bool,
    ) -> Result<DecodedBlock> {
        let verify_lookup = self.needs_verification(
            block_index,
            verify_checksum,
            BlockVerification::LookupVerified,
        );
        let block = read_block(
            &self.file,
            &self.block_index,
            block_index,
            Self::block_read_options(geometry, verify_lookup),
        )?;
        if verify_lookup {
            self.mark_verified(block_index, BlockVerification::LookupVerified);
        }
        Ok(block)
    }

    /// Reads and decodes a block while reusing the caller-owned backing buffer.
    pub(crate) fn load_block_reusing(
        &self,
        block_index: usize,
        geometry: SegmentGeometry,
        verify_checksum: bool,
        buffer: Vec<u8>,
    ) -> Result<DecodedBlock> {
        let verify_lookup = self.needs_verification(
            block_index,
            verify_checksum,
            BlockVerification::LookupVerified,
        );
        let block = read_block_reusing(
            &self.file,
            &self.block_index,
            block_index,
            Self::block_read_options(geometry, verify_lookup),
            buffer,
        )?;
        if verify_lookup {
            self.mark_verified(block_index, BlockVerification::LookupVerified);
        }
        Ok(block)
    }

    pub(crate) fn verify_block_payload(
        &self,
        block_index: usize,
        block: &DecodedBlock,
        verify_checksum: bool,
    ) -> Result<()> {
        if self.needs_verification(
            block_index,
            verify_checksum,
            BlockVerification::PayloadVerified,
        ) {
            block.verify_payload_checksum()?;
            self.mark_verified(block_index, BlockVerification::PayloadVerified);
        }
        Ok(())
    }

    fn block_read_options(
        geometry: SegmentGeometry,
        verify_lookup_checksum: bool,
    ) -> BlockReadOptions {
        BlockReadOptions {
            geometry,
            verify_lookup_checksum,
        }
    }

    fn needs_verification(
        &self,
        block_index: usize,
        requested: bool,
        required: BlockVerification,
    ) -> bool {
        requested
            && self
                .block_verifications
                .lock()
                .requires(block_index, required)
    }

    fn mark_verified(&self, block_index: usize, verified: BlockVerification) {
        self.block_verifications
            .lock()
            .advance(block_index, verified);
    }
}

impl BlockVerificationCache {
    fn new(block_count: usize) -> Self {
        Self {
            phases: vec![BlockVerification::Unverified; block_count],
        }
    }

    fn requires(&self, block_index: usize, required: BlockVerification) -> bool {
        self.phases[block_index] < required
    }

    fn advance(&mut self, block_index: usize, verified: BlockVerification) {
        let current = self.phases[block_index];
        debug_assert!(
            verified != BlockVerification::PayloadVerified
                || current >= BlockVerification::LookupVerified,
            "payload verification requires trusted lookup metadata"
        );
        self.phases[block_index] = current.max(verified);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verification_advances_independently_per_block() {
        let mut blocks = BlockVerificationCache::new(2);

        assert!(blocks.requires(0, BlockVerification::LookupVerified));
        assert!(blocks.requires(0, BlockVerification::PayloadVerified));

        blocks.advance(0, BlockVerification::LookupVerified);
        assert!(!blocks.requires(0, BlockVerification::LookupVerified));
        assert!(blocks.requires(0, BlockVerification::PayloadVerified));
        assert!(blocks.requires(1, BlockVerification::LookupVerified));

        blocks.advance(0, BlockVerification::PayloadVerified);
        assert!(!blocks.requires(0, BlockVerification::PayloadVerified));
    }
}
