//! Runtime state shared by the public `Store`, lookup sessions, and cursors.

use std::{fs::File, sync::Arc};

use parking_lot::{Mutex, RwLock};

use crate::{
    engine::{
        io::WriterLock,
        paths::StorePaths,
        segment_file::{
            OpenedSegment, read_block, read_block_metadata_reusing, read_block_payload,
            read_block_reusing,
        },
    },
    error::Result,
    format::{
        ValueLayout, block::DecodedBlock, manifest::StoreManifest, segment::BlockIndexEntry,
        store_file::StoreDescriptor,
    },
};

pub(crate) struct StoreInner {
    pub(crate) paths: StorePaths,
    pub(crate) geometry: StoreGeometry,
    pub(crate) verify_block_checksums: bool,
    pub(crate) commit_lock: Mutex<()>,
    pub(crate) state: RwLock<StoreState>,
    /// Held for the store's lifetime by a writer open; `None` for read-only opens.
    pub(crate) writer_lock: Option<WriterLock>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct StoreGeometry {
    pub(crate) key_len: usize,
    pub(crate) value_layout: ValueLayout,
}

pub(crate) struct StoreState {
    pub(crate) manifest: StoreManifest,
    pub(crate) main_segments: SegmentSnapshot,
    pub(crate) patch_segments: SegmentSnapshot,
}

pub(crate) type SegmentSnapshot = Arc<[Arc<SegmentState>]>;

/// Visible immutable segment and its in-memory sparse block index.
pub(crate) struct SegmentState {
    pub(crate) segment_id: u32,
    pub(crate) file: File,
    pub(crate) min_key: Vec<u8>,
    pub(crate) max_key: Vec<u8>,
    pub(crate) block_index: Vec<BlockIndexEntry>,
}

impl StoreGeometry {
    pub(crate) fn from_descriptor(descriptor: &StoreDescriptor) -> Self {
        Self {
            key_len: descriptor.key_len,
            value_layout: descriptor.value_layout,
        }
    }
}

impl StoreState {
    pub(crate) fn new(
        manifest: StoreManifest,
        main_segments: Vec<Arc<SegmentState>>,
        patch_segments: Vec<Arc<SegmentState>>,
    ) -> Self {
        Self {
            manifest,
            main_segments: main_segments.into(),
            patch_segments: patch_segments.into(),
        }
    }

    pub(crate) fn main_segments_as_vec(&self) -> Vec<Arc<SegmentState>> {
        self.main_segments.iter().cloned().collect()
    }

    pub(crate) fn patch_segments_as_vec(&self) -> Vec<Arc<SegmentState>> {
        self.patch_segments.iter().cloned().collect()
    }

    pub(crate) fn replace_segments(
        &mut self,
        main_segments: Vec<Arc<SegmentState>>,
        patch_segments: Vec<Arc<SegmentState>>,
    ) {
        self.main_segments = main_segments.into();
        self.patch_segments = patch_segments.into();
    }
}

impl SegmentState {
    /// Converts a verified on-disk segment into the in-memory state used by readers.
    pub(super) fn from_opened(segment_id: u32, opened: OpenedSegment) -> Self {
        Self {
            segment_id,
            file: opened.file,
            min_key: opened.min_key,
            max_key: opened.max_key,
            block_index: opened.block_index,
        }
    }

    /// Builds runtime state directly from newly written segment metadata.
    pub(crate) fn from_written(
        segment_id: u32,
        file: File,
        min_key: Vec<u8>,
        max_key: Vec<u8>,
        block_index: Vec<BlockIndexEntry>,
    ) -> Self {
        Self {
            segment_id,
            file,
            min_key,
            max_key,
            block_index,
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

    /// Reads and decodes only the key/index metadata for a block.
    pub(crate) fn load_block_metadata_reusing(
        &self,
        block_index: usize,
        key_len: usize,
        value_layout: ValueLayout,
        verify_checksum: bool,
        buffer: Vec<u8>,
    ) -> Result<DecodedBlock> {
        let entry = &self.block_index[block_index];
        read_block_metadata_reusing(
            &self.file,
            entry,
            key_len,
            value_layout,
            verify_checksum,
            buffer,
        )
    }

    /// Loads the value payload for a metadata-only decoded block.
    pub(crate) fn load_block_payload(
        &self,
        block_index: usize,
        block: &mut DecodedBlock,
        verify_checksum: bool,
    ) -> Result<()> {
        let entry = &self.block_index[block_index];
        read_block_payload(&self.file, entry, block, verify_checksum)
    }
}
