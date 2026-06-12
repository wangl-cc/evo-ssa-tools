//! File-level access to immutable segment files: open-and-validate, block
//! reads.
//!
//! All decoding is delegated to [`crate::format`]; this module reads the right
//! byte ranges from disk and applies the open-time validation policy (a
//! segment that fails any check is treated as absent, never as an error).

use std::{fs::File, path::PathBuf};

use crate::{
    engine::io::read_exact_at,
    error::Result,
    format::{
        ValueLayout,
        block::DecodedBlock,
        segment::{
            BlockIndexEntry, SEGMENT_FOOTER_TRAILER_LEN, SEGMENT_HEADER_LEN, SegmentFooter,
            SegmentHeader,
        },
    },
};

#[derive(Clone, Copy)]
pub(crate) struct SegmentOpenOptions {
    pub(crate) expected_key_len: usize,
    pub(crate) expected_value_layout: ValueLayout,
}

/// Open segment handle with its sparse block index loaded into memory.
#[derive(Debug)]
pub(crate) struct OpenedSegment {
    pub(crate) file: File,
    pub(crate) min_key: Vec<u8>,
    pub(crate) max_key: Vec<u8>,
    pub(crate) block_index: Vec<BlockIndexEntry>,
}

impl OpenedSegment {
    /// Opens and validates one segment file.
    ///
    /// Returns `Ok(None)` when the file is missing, malformed, corrupt, or
    /// does not match the expected store geometry: a referenced segment that
    /// fails to open is miss space, not an error.
    pub(crate) fn open(path: PathBuf, options: SegmentOpenOptions) -> Result<Option<Self>> {
        let file = match File::open(&path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let header = match read_header(&file) {
            Ok(header) => header,
            Err(_) => return Ok(None),
        };
        if !header.matches_geometry(options.expected_key_len, options.expected_value_layout) {
            return Ok(None);
        }
        let footer = match read_footer(&file, options.expected_key_len) {
            Ok(footer) => footer,
            Err(_) => return Ok(None),
        };
        Ok(Some(Self {
            file,
            min_key: footer.min_key,
            max_key: footer.max_key,
            block_index: footer.block_index,
        }))
    }
}

fn read_header(file: &File) -> Result<SegmentHeader> {
    let mut bytes = [0u8; SEGMENT_HEADER_LEN];
    read_exact_at(file, 0, &mut bytes)?;
    Ok(SegmentHeader::from_bytes(&bytes)?)
}

fn read_footer(file: &File, key_len: usize) -> Result<SegmentFooter> {
    use crate::format::CorruptionError;

    let file_len = file.metadata()?.len();
    let trailer_len = SEGMENT_FOOTER_TRAILER_LEN as u64;
    if file_len < trailer_len {
        return Err(CorruptionError::SegmentFormat.into());
    }
    let mut trailer = [0u8; SEGMENT_FOOTER_TRAILER_LEN];
    read_exact_at(file, file_len - trailer_len, &mut trailer)?;
    let footer_body_len = u64::from(u32::from_le_bytes(
        *trailer.first_chunk::<4>().expect("trailer is 8 bytes"),
    ));
    let footer_len = footer_body_len
        .checked_add(trailer_len)
        .ok_or(CorruptionError::SegmentFormat)?;
    if file_len < footer_len {
        return Err(CorruptionError::SegmentFormat.into());
    }
    let data_end = file_len - footer_len;
    let footer_len = usize::try_from(footer_len).map_err(|_| CorruptionError::SegmentFormat)?;
    let mut bytes = vec![0u8; footer_len];
    read_exact_at(file, data_end, &mut bytes)?;
    Ok(SegmentFooter::from_bytes(&bytes, key_len, data_end)?)
}

/// Reads and decodes one block, allocating a fresh buffer.
pub(crate) fn read_block(
    file: &File,
    entry: &BlockIndexEntry,
    key_len: usize,
    value_layout: ValueLayout,
    verify_checksum: bool,
) -> Result<DecodedBlock> {
    read_block_reusing(
        file,
        entry,
        key_len,
        value_layout,
        verify_checksum,
        Vec::new(),
    )
}

/// Reads and decodes one block while reusing a caller-owned backing buffer.
pub(crate) fn read_block_reusing(
    file: &File,
    entry: &BlockIndexEntry,
    key_len: usize,
    value_layout: ValueLayout,
    verify_checksum: bool,
    mut bytes: Vec<u8>,
) -> Result<DecodedBlock> {
    let block_len = entry.block_len as usize;
    if bytes.len() < block_len {
        bytes.resize(block_len, 0);
    } else {
        bytes.truncate(block_len);
    }
    read_exact_at(file, entry.block_offset, &mut bytes)?;
    Ok(DecodedBlock::decode(
        bytes,
        entry,
        key_len,
        value_layout,
        verify_checksum,
    )?)
}
