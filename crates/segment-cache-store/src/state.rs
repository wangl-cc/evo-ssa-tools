#[cfg(test)]
use std::path::PathBuf;
use std::{fs::File, io::ErrorKind, sync::Arc};

use parking_lot::RwLock;

use crate::{
    error::{Error, Result},
    format::{BlockIndexEntry, DecodedBlock, OpenedSegment, read_block, read_block_reusing},
    manifest::StoreManifest,
    options::{StoreOptions, ValueLayout},
};

pub(crate) struct StoreInner {
    pub(crate) options: StoreOptions,
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

pub(crate) struct SegmentState {
    #[cfg(test)]
    pub(crate) path: PathBuf,
    pub(crate) file: File,
    pub(crate) min_key: Vec<u8>,
    pub(crate) max_key: Vec<u8>,
    pub(crate) block_index: Vec<BlockIndexEntry>,
}

pub(crate) fn segment_state_from_opened(opened: OpenedSegment) -> SegmentState {
    SegmentState {
        #[cfg(test)]
        path: opened.path,
        file: opened.file,
        min_key: opened.min_key,
        max_key: opened.max_key,
        block_index: opened.block_index,
    }
}

pub(crate) fn find_block_index(segment: &SegmentState, key: &[u8]) -> usize {
    let idx = segment
        .block_index
        .partition_point(|entry| entry.first_key.as_slice() <= key);
    idx.saturating_sub(1)
}

pub(crate) fn load_segment_block(
    segment: &SegmentState,
    block_index: usize,
    key_len: usize,
    value_layout: ValueLayout,
    verify_checksum: bool,
) -> Result<DecodedBlock> {
    let entry = &segment.block_index[block_index];
    read_block(&segment.file, entry, key_len, value_layout, verify_checksum)
}

pub(crate) fn load_segment_block_reusing(
    segment: &SegmentState,
    block_index: usize,
    key_len: usize,
    value_layout: ValueLayout,
    verify_checksum: bool,
    buffer: Vec<u8>,
) -> Result<DecodedBlock> {
    let entry = &segment.block_index[block_index];
    read_block_reusing(
        &segment.file,
        entry,
        key_len,
        value_layout,
        verify_checksum,
        buffer,
    )
}

pub(crate) fn is_block_miss_error(error: &Error) -> bool {
    match error {
        Error::CorruptBlock => true,
        Error::UnsupportedFormatVersion { .. } => true,
        Error::Io(error) if error.kind() == ErrorKind::UnexpectedEof => true,
        _ => false,
    }
}
