//! Open immutable segment state and process-local block verification cache.

use std::{cmp::Ordering, fs::File};

#[cfg(feature = "block-checksum")]
use parking_lot::Mutex;

use super::{
    SegmentGeometry,
    file::{BlockReadOptions, OpenedSegment, read_block, read_block_reusing},
    index::BlockIndexEntry,
    writer::SegmentFileMetadata,
};
use crate::{block::DecodedBlock, error::Result};

/// Visible immutable segment and its in-memory sparse block index.
pub(crate) struct Segment {
    segment_id: u32,
    file: File,
    min_key: Vec<u8>,
    max_key: Vec<u8>,
    block_index: Vec<BlockIndexEntry>,
    #[cfg(feature = "block-checksum")]
    verified_blocks: Mutex<VerifiedBlocks>,
}

/// Process-local verification state for immutable segment blocks.
///
/// Once a block passes checksum verification, later reads through the same
/// open segment can skip that work. Segment files are immutable while open, so
/// dropping the [`Segment`] is the only invalidation required.
#[derive(Debug)]
#[cfg(feature = "block-checksum")]
struct VerifiedBlocks {
    blocks: Vec<bool>,
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

    pub(crate) fn block_contains(&self, index: usize, key: &[u8]) -> bool {
        self.block_index[index].key_range.contains(key)
    }

    pub(crate) fn block_min_cmp(&self, index: usize, key: &[u8]) -> Ordering {
        self.block_index[index].key_range.min_cmp(key)
    }

    pub(crate) fn block_max_cmp(&self, index: usize, key: &[u8]) -> Ordering {
        self.block_index[index].key_range.max_cmp(key)
    }

    /// Converts a verified on-disk segment into the state used by readers.
    pub(super) fn from_opened(segment_id: u32, opened: OpenedSegment) -> Self {
        #[cfg(feature = "block-checksum")]
        let verified_blocks = VerifiedBlocks::new(opened.block_index.len());
        Self {
            segment_id,
            file: opened.file,
            min_key: opened.min_key,
            max_key: opened.max_key,
            block_index: opened.block_index,
            #[cfg(feature = "block-checksum")]
            verified_blocks: Mutex::new(verified_blocks),
        }
    }

    /// Builds state directly from newly written segment metadata.
    pub(crate) fn from_written(segment_id: u32, file: File, metadata: SegmentFileMetadata) -> Self {
        let min_key = metadata.min_key;
        let max_key = metadata.max_key;
        let footer = metadata.footer;
        let block_index = footer.block_index;
        #[cfg(feature = "block-checksum")]
        let verified_blocks = VerifiedBlocks::new(block_index.len());
        Self {
            segment_id,
            file,
            min_key,
            max_key,
            block_index,
            #[cfg(feature = "block-checksum")]
            verified_blocks: Mutex::new(verified_blocks),
        }
    }

    /// Finds the sparse block-index entry that may contain `key`.
    pub(crate) fn find_block_index(&self, key: &[u8]) -> usize {
        let idx = self
            .block_index
            .partition_point(|entry| entry.key_range.min_cmp(key) != Ordering::Greater);
        idx.saturating_sub(1)
    }

    /// Reads and decodes a block by sparse block-index position.
    pub(crate) fn load_block(
        &self,
        block_index: usize,
        geometry: SegmentGeometry,
        verify_checksum: bool,
    ) -> Result<DecodedBlock> {
        let entry = &self.block_index[block_index];
        let verify = self.needs_verification(block_index, verify_checksum);
        let block = read_block(
            &self.file,
            entry,
            Self::block_read_options(geometry, verify),
        )?;
        #[cfg(feature = "block-checksum")]
        if verify {
            self.mark_verified(block_index);
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
        let entry = &self.block_index[block_index];
        let verify = self.needs_verification(block_index, verify_checksum);
        let block = read_block_reusing(
            &self.file,
            entry,
            Self::block_read_options(geometry, verify),
            buffer,
        )?;
        #[cfg(feature = "block-checksum")]
        if verify {
            self.mark_verified(block_index);
        }
        Ok(block)
    }

    fn block_read_options(geometry: SegmentGeometry, _verify_checksum: bool) -> BlockReadOptions {
        BlockReadOptions {
            geometry,
            #[cfg(feature = "block-checksum")]
            verify_checksum: _verify_checksum,
        }
    }

    #[cfg(feature = "block-checksum")]
    fn needs_verification(&self, block_index: usize, requested: bool) -> bool {
        requested && !self.verified_blocks.lock().is_verified(block_index)
    }

    #[cfg(not(feature = "block-checksum"))]
    fn needs_verification(&self, _block_index: usize, _requested: bool) -> bool {
        false
    }

    #[cfg(feature = "block-checksum")]
    fn mark_verified(&self, block_index: usize) {
        self.verified_blocks.lock().mark(block_index);
    }
}

#[cfg(feature = "block-checksum")]
impl VerifiedBlocks {
    fn new(block_count: usize) -> Self {
        Self {
            blocks: vec![false; block_count],
        }
    }

    fn is_verified(&self, block_index: usize) -> bool {
        self.blocks[block_index]
    }

    fn mark(&mut self, block_index: usize) {
        self.blocks[block_index] = true;
    }
}

#[cfg(all(test, feature = "block-checksum"))]
mod tests {
    use super::*;

    #[test]
    fn full_block_verification_is_tracked_per_block() {
        let mut blocks = VerifiedBlocks::new(2);

        assert!(!blocks.is_verified(0));
        blocks.mark(0);
        assert!(blocks.is_verified(0));
        assert!(!blocks.is_verified(1));
    }
}
