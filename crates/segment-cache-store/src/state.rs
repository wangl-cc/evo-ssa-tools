//! Runtime state shared by the public `Store`, lookup sessions, and cursors.

use std::{fs::File, sync::Arc};

use parking_lot::{Mutex, RwLock};

use crate::{
    error::Result,
    manifest::StoreManifest,
    options::{StoreOptions, ValueLayout},
    segment::format::{
        BlockIndexEntry, DecodedBlock, OpenedSegment, read_block, read_block_reusing,
    },
};

pub(crate) struct StoreInner {
    pub(crate) options: StoreOptions,
    pub(crate) commit_lock: Mutex<()>,
    pub(crate) state: RwLock<StoreState>,
}

pub(crate) struct StoreState {
    pub(crate) manifest: StoreManifest,
    pub(crate) shards: Vec<ShardState>,
}

pub(crate) struct ShardState {
    pub(crate) segments: Vec<Arc<SegmentState>>,
    pub(crate) last_max_key: Option<Vec<u8>>,
}

/// Visible immutable segment and its in-memory sparse block index.
pub(crate) struct SegmentState {
    pub(crate) file: File,
    pub(crate) min_key: Vec<u8>,
    pub(crate) max_key: Vec<u8>,
    pub(crate) block_index: Vec<BlockIndexEntry>,
}

impl SegmentState {
    /// Converts a verified on-disk segment into the in-memory state used by readers.
    pub(crate) fn from_opened(opened: OpenedSegment) -> Self {
        Self {
            file: opened.file,
            min_key: opened.min_key,
            max_key: opened.max_key,
            block_index: opened.block_index,
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
        key_len: usize,
        value_layout: ValueLayout,
        verify_checksum: bool,
    ) -> Result<DecodedBlock> {
        let entry = &self.block_index[block_index];
        read_block(&self.file, entry, key_len, value_layout, verify_checksum)
    }

    /// Reads and decodes a block while reusing the caller-owned backing buffer.
    pub(crate) fn load_block_reusing(
        &self,
        block_index: usize,
        key_len: usize,
        value_layout: ValueLayout,
        verify_checksum: bool,
        buffer: Vec<u8>,
    ) -> Result<DecodedBlock> {
        let entry = &self.block_index[block_index];
        read_block_reusing(
            &self.file,
            entry,
            key_len,
            value_layout,
            verify_checksum,
            buffer,
        )
    }
}
