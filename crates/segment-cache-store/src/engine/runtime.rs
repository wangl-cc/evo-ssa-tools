//! Runtime state shared by the public `Store`, lookup sessions, and cursors.

use std::{fs::File, sync::Arc};

use parking_lot::{Mutex, RwLock};

use crate::{
    engine::{
        io::WriterLock,
        paths::StorePaths,
        segment_file::{
            BlockReadOptions, OpenedSegment, read_block, read_block_metadata_reusing,
            read_block_payload, read_block_reusing,
        },
    },
    error::Result,
    format::{
        BlockChecksumKind, ValueLayout, ValuePayloadCompressionKind, block::DecodedBlock,
        manifest::StoreManifest, segment::BlockIndexEntry, store_file::StoreDescriptor,
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
    pub(crate) block_checksum: BlockChecksumKind,
    pub(crate) value_payload_compression: ValuePayloadCompressionKind,
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
    verified_blocks: Mutex<VerifiedBlocks>,
}

/// Process-local verification state for immutable segment blocks.
///
/// A block that has already passed checksum verification can skip repeated
/// checksum work on later reads from the same open store. Segment files are
/// immutable while a store is open, so the cache needs no invalidation beyond
/// dropping the [`SegmentState`].
#[derive(Debug)]
struct VerifiedBlocks {
    blocks: Vec<VerifiedBlock>,
}

#[derive(Clone, Copy, Debug)]
enum VerifiedBlockPart {
    Metadata,
    Payload,
    Full,
}

#[derive(Clone, Copy, Debug, Default)]
struct VerifiedBlock {
    metadata: bool,
    payload: bool,
}

impl StoreGeometry {
    pub(crate) fn from_descriptor(
        descriptor: &StoreDescriptor,
        block_checksum: BlockChecksumKind,
        value_payload_compression: ValuePayloadCompressionKind,
    ) -> Self {
        Self {
            key_len: descriptor.key_len,
            value_layout: descriptor.value_layout,
            block_checksum,
            value_payload_compression,
        }
    }

    fn block_read_options(self, verify_checksum: bool) -> BlockReadOptions {
        BlockReadOptions {
            key_len: self.key_len,
            value_layout: self.value_layout,
            block_checksum: self.block_checksum,
            value_payload_compression: self.value_payload_compression,
            verify_checksum,
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

    /// Builds runtime state directly from newly written segment metadata.
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
        let verify = self.needs_verification(block_index, VerifiedBlockPart::Full, verify_checksum);
        let block = read_block(&self.file, entry, geometry.block_read_options(verify))?;
        if verify_checksum {
            self.mark_verified(block_index, VerifiedBlockPart::Full);
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
        let verify = self.needs_verification(block_index, VerifiedBlockPart::Full, verify_checksum);
        let block = read_block_reusing(
            &self.file,
            entry,
            geometry.block_read_options(verify),
            buffer,
        )?;
        if verify_checksum {
            self.mark_verified(block_index, VerifiedBlockPart::Full);
        }
        Ok(block)
    }

    /// Reads and decodes only the key/index metadata for a block.
    pub(crate) fn load_block_metadata_reusing(
        &self,
        block_index: usize,
        geometry: StoreGeometry,
        verify_checksum: bool,
        buffer: Vec<u8>,
    ) -> Result<DecodedBlock> {
        let entry = &self.block_index[block_index];
        let verify =
            self.needs_verification(block_index, VerifiedBlockPart::Metadata, verify_checksum);
        let block = read_block_metadata_reusing(
            &self.file,
            entry,
            geometry.block_read_options(verify),
            buffer,
        )?;
        if verify_checksum {
            self.mark_verified(block_index, VerifiedBlockPart::Metadata);
        }
        Ok(block)
    }

    /// Loads the value payload for a metadata-only decoded block.
    pub(crate) fn load_block_payload(
        &self,
        block_index: usize,
        block: &mut DecodedBlock,
        verify_checksum: bool,
    ) -> Result<()> {
        let entry = &self.block_index[block_index];
        let verify =
            self.needs_verification(block_index, VerifiedBlockPart::Payload, verify_checksum);
        read_block_payload(&self.file, entry, block, verify)?;
        if verify_checksum {
            self.mark_verified(block_index, VerifiedBlockPart::Payload);
        }
        Ok(())
    }

    fn needs_verification(
        &self,
        block_index: usize,
        part: VerifiedBlockPart,
        requested: bool,
    ) -> bool {
        requested && !self.verified_blocks.lock().is_verified(block_index, part)
    }

    fn mark_verified(&self, block_index: usize, part: VerifiedBlockPart) {
        self.verified_blocks.lock().mark(block_index, part);
    }
}

impl VerifiedBlocks {
    fn new(block_count: usize) -> Self {
        Self {
            blocks: vec![VerifiedBlock::default(); block_count],
        }
    }

    fn is_verified(&self, block_index: usize, part: VerifiedBlockPart) -> bool {
        self.blocks
            .get(block_index)
            .is_some_and(|block| match part {
                VerifiedBlockPart::Metadata => block.metadata,
                VerifiedBlockPart::Payload => block.payload,
                VerifiedBlockPart::Full => block.metadata && block.payload,
            })
    }

    fn mark(&mut self, block_index: usize, part: VerifiedBlockPart) {
        if let Some(block) = self.blocks.get_mut(block_index) {
            match part {
                VerifiedBlockPart::Metadata => block.metadata = true,
                VerifiedBlockPart::Payload => block.payload = true,
                VerifiedBlockPart::Full => {
                    block.metadata = true;
                    block.payload = true;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod verified_blocks {
        use super::*;

        #[test]
        fn metadata_and_payload_are_tracked_independently() {
            let mut blocks = VerifiedBlocks::new(2);

            assert!(!blocks.is_verified(0, VerifiedBlockPart::Full));
            blocks.mark(0, VerifiedBlockPart::Metadata);
            assert!(blocks.is_verified(0, VerifiedBlockPart::Metadata));
            assert!(!blocks.is_verified(0, VerifiedBlockPart::Payload));
            assert!(!blocks.is_verified(0, VerifiedBlockPart::Full));

            blocks.mark(0, VerifiedBlockPart::Payload);
            assert!(blocks.is_verified(0, VerifiedBlockPart::Full));
            assert!(!blocks.is_verified(1, VerifiedBlockPart::Metadata));
        }

        #[test]
        fn out_of_range_marks_are_ignored() {
            let mut blocks = VerifiedBlocks::new(1);

            blocks.mark(9, VerifiedBlockPart::Full);

            assert!(!blocks.is_verified(0, VerifiedBlockPart::Full));
            assert!(!blocks.is_verified(9, VerifiedBlockPart::Full));
        }
    }
}
