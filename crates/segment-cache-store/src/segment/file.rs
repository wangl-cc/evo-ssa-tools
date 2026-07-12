//! File-level access to immutable segment files: open-and-validate, block
//! reads.
//!
//! Block and segment codecs own byte interpretation; this module reads the
//! required ranges and applies the open-time validation policy (a segment that
//! fails any check is treated as absent, never as an error).

use std::{fs::File, path::PathBuf};

use super::{
    Segment, SegmentFingerprint, SegmentGeometry,
    format::{SEGMENT_FOOTER_TRAILER_LEN, SEGMENT_HEADER_LEN, SegmentFooter, SegmentHeader},
    index::BlockIndexEntry,
    io::read_exact_at,
};
use crate::{
    binary::BinaryCursor,
    block::{BlockDecodeOptions, BlockKeyRangeRef, DecodedBlock},
    error::{CorruptionError, Result},
};

#[derive(Clone, Copy)]
pub(crate) struct SegmentOpenOptions<'a> {
    pub(crate) geometry: SegmentGeometry,
    pub(crate) expected_fingerprint: SegmentFingerprint,
    pub(crate) expected_min_key: &'a [u8],
    pub(crate) expected_max_key: &'a [u8],
}

#[derive(Clone, Copy)]
pub(crate) struct BlockReadOptions {
    pub(crate) geometry: SegmentGeometry,
    #[cfg(feature = "block-checksum")]
    pub(crate) verify_checksum: bool,
}

/// Open segment handle with its sparse block index loaded into memory.
#[derive(Debug)]
pub(super) struct OpenedSegment {
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
    pub(crate) fn open(path: PathBuf, options: SegmentOpenOptions<'_>) -> Result<Option<Self>> {
        let file = match File::open(&path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let header = match read_header(&file) {
            Ok(header) => header,
            Err(error) if error.is_cache_miss_corruption() => return Ok(None),
            Err(error) => return Err(error),
        };
        if !header.matches_geometry(
            options.geometry.key_len,
            options.geometry.value_layout,
            options.geometry.block_checksum,
            options.geometry.value_payload_compression,
        ) {
            return Ok(None);
        }
        let footer = match read_footer(
            &file,
            options.geometry.key_len,
            options.expected_min_key,
            options.expected_max_key,
        ) {
            Ok(footer) => footer,
            Err(error) if error.is_cache_miss_corruption() => return Ok(None),
            Err(error) => return Err(error),
        };
        let fingerprint = match SegmentFingerprint::from_file(&file) {
            Ok(fingerprint) => fingerprint,
            Err(error) if error.is_cache_miss_corruption() => return Ok(None),
            Err(error) => return Err(error),
        };
        if fingerprint != options.expected_fingerprint {
            return Ok(None);
        }
        Ok(Some(Self {
            file,
            min_key: options.expected_min_key.to_vec(),
            max_key: options.expected_max_key.to_vec(),
            block_index: footer.block_index,
        }))
    }
}

impl Segment {
    /// Opens and validates one manifest-referenced segment.
    pub(crate) fn open(
        segment_id: u32,
        path: PathBuf,
        options: SegmentOpenOptions<'_>,
    ) -> Result<Option<Self>> {
        Ok(OpenedSegment::open(path, options)?.map(|opened| Self::from_opened(segment_id, opened)))
    }
}

fn read_header(file: &File) -> Result<SegmentHeader> {
    let mut bytes = [0u8; SEGMENT_HEADER_LEN];
    read_exact_at(file, 0, &mut bytes)?;
    Ok(SegmentHeader::from_bytes(&bytes)?)
}

fn read_footer(
    file: &File,
    key_len: usize,
    expected_min_key: &[u8],
    expected_max_key: &[u8],
) -> Result<SegmentFooter> {
    use crate::error::CorruptionError;

    let file_len = file.metadata()?.len();
    let trailer_len = SEGMENT_FOOTER_TRAILER_LEN as u64;
    if file_len < trailer_len {
        return Err(CorruptionError::SegmentFormat.into());
    }
    let mut trailer = [0u8; SEGMENT_FOOTER_TRAILER_LEN];
    read_exact_at(file, file_len - trailer_len, &mut trailer)?;
    let footer_body_len = u64::from(
        BinaryCursor::new(&trailer)
            .read::<u32>()
            .ok_or(CorruptionError::SegmentFormat)?,
    );
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
    Ok(SegmentFooter::from_bytes(
        &bytes,
        key_len,
        data_end,
        expected_min_key,
        expected_max_key,
    )?)
}

/// Reads and decodes one block, allocating a fresh buffer.
pub(super) fn read_block(
    file: &File,
    entry: &BlockIndexEntry,
    options: BlockReadOptions,
) -> Result<DecodedBlock> {
    read_block_reusing(file, entry, options, Vec::new())
}

/// Reads and decodes one block while reusing a caller-owned backing buffer.
pub(super) fn read_block_reusing(
    file: &File,
    entry: &BlockIndexEntry,
    options: BlockReadOptions,
    mut bytes: Vec<u8>,
) -> Result<DecodedBlock> {
    let block_len = usize::try_from(entry.byte_range.end - entry.byte_range.start)
        .map_err(|_| CorruptionError::SegmentFormat)?;
    if bytes.len() < block_len {
        bytes.resize(block_len, 0);
    } else {
        bytes.truncate(block_len);
    }
    read_exact_at(file, entry.byte_range.start, &mut bytes)?;
    Ok(DecodedBlock::decode(bytes, BlockDecodeOptions {
        expected_key_range: BlockKeyRangeRef {
            prefix: entry.key_range.prefix(),
            min_suffix: entry.key_range.min_suffix(),
            max_suffix: entry.key_range.max_suffix(),
        },
        key_len: options.geometry.key_len,
        value_layout: options.geometry.value_layout,
        block_checksum: options.geometry.block_checksum,
        value_payload_compression: options.geometry.value_payload_compression,
        #[cfg(feature = "block-checksum")]
        verify_checksum: options.verify_checksum,
    })?)
}
