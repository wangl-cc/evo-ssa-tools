//! Open immutable segment state and process-local block verification cache.

use std::fs::File;

use parking_lot::Mutex;

use crate::{
    block::{BlockKeyUpperBound, DecodedBlock},
    error::Result,
    schema::StoreGeometry,
    segment::{
        file::{BlockReadOptions, OpenedSegment, read_block, read_block_reusing},
        format::BlockIndexEntry,
    },
};

/// Visible immutable segment and its in-memory sparse block index.
pub(crate) struct SegmentState {
    pub(crate) segment_id: u32,
    pub(crate) file: File,
    pub(crate) min_key: Vec<u8>,
    pub(crate) max_key: Vec<u8>,
    pub(crate) block_index: Vec<BlockIndexEntry>,
    verified_blocks: Mutex<VerifiedBlocks>,
}

/// Process-local verification state for immutable segment blocks.
///
/// Once a block passes checksum verification, later reads through the same
/// open segment can skip that work. Segment files are immutable while open, so
/// dropping the [`SegmentState`] is the only invalidation required.
#[derive(Debug)]
struct VerifiedBlocks {
    blocks: Vec<bool>,
}

impl SegmentState {
    /// Converts a verified on-disk segment into the state used by readers.
    pub(crate) fn from_opened(segment_id: u32, opened: OpenedSegment) -> Self {
        let verified_blocks = VerifiedBlocks::new(opened.block_index.len());
        Self {
            segment_id,
            file: opened.file,
            min_key: opened.min_key,
            max_key: opened.max_key,
            block_index: opened.block_index,
            verified_blocks: Mutex::new(verified_blocks),
        }
    }

    /// Builds state directly from newly written segment metadata.
    pub(crate) fn from_written(
        segment_id: u32,
        file: File,
        min_key: Vec<u8>,
        max_key: Vec<u8>,
        block_index: Vec<BlockIndexEntry>,
    ) -> Self {
        let verified_blocks = VerifiedBlocks::new(block_index.len());
        Self {
            segment_id,
            file,
            min_key,
            max_key,
            block_index,
            verified_blocks: Mutex::new(verified_blocks),
        }
    }

    /// Finds the sparse block-index entry that may contain `key`.
    pub(crate) fn find_block_index(&self, key: &[u8]) -> usize {
        let idx = self
            .block_index
            .partition_point(|entry| entry.first_key.as_slice() <= key);
        idx.saturating_sub(1)
    }

    /// Reads and decodes a block by sparse block-index position.
    pub(crate) fn load_block(
        &self,
        block_index: usize,
        geometry: StoreGeometry,
        verify_checksum: bool,
    ) -> Result<DecodedBlock> {
        let entry = &self.block_index[block_index];
        let verify = self.needs_verification(block_index, verify_checksum);
        let block = read_block(
            &self.file,
            entry,
            Self::block_read_options(geometry, verify, self.block_upper_key_bound(block_index)),
        )?;
        if verify_checksum {
            self.mark_verified(block_index);
        }
        Ok(block)
    }

    /// Reads and decodes a block while reusing the caller-owned backing buffer.
    pub(crate) fn load_block_reusing(
        &self,
        block_index: usize,
        geometry: StoreGeometry,
        verify_checksum: bool,
        buffer: Vec<u8>,
    ) -> Result<DecodedBlock> {
        let entry = &self.block_index[block_index];
        let verify = self.needs_verification(block_index, verify_checksum);
        let block = read_block_reusing(
            &self.file,
            entry,
            Self::block_read_options(geometry, verify, self.block_upper_key_bound(block_index)),
            buffer,
        )?;
        if verify_checksum {
            self.mark_verified(block_index);
        }
        Ok(block)
    }

    fn block_read_options<'a>(
        geometry: StoreGeometry,
        verify_checksum: bool,
        upper_key_bound: BlockKeyUpperBound<'a>,
    ) -> BlockReadOptions<'a> {
        BlockReadOptions {
            key_len: geometry.key_len,
            value_layout: geometry.value_layout,
            block_checksum: geometry.block_checksum,
            value_payload_compression: geometry.value_payload_compression,
            verify_checksum,
            upper_key_bound,
        }
    }

    fn needs_verification(&self, block_index: usize, requested: bool) -> bool {
        requested && !self.verified_blocks.lock().is_verified(block_index)
    }

    fn block_upper_key_bound(&self, block_index: usize) -> BlockKeyUpperBound<'_> {
        match self.block_index.get(block_index + 1) {
            Some(next) => BlockKeyUpperBound::Exclusive(next.first_key.as_slice()),
            None => BlockKeyUpperBound::Inclusive(self.max_key.as_slice()),
        }
    }

    fn mark_verified(&self, block_index: usize) {
        self.verified_blocks.lock().mark(block_index);
    }
}

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

#[cfg(test)]
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
