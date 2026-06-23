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
        BlockChecksumKind, ValueLayout, ValuePayloadCompressionKind,
        block::{BlockDecodeOptions, BlockKeyUpperBound, DecodedBlock},
        manifest::SegmentFileFingerprint,
        segment::{
            BlockIndexEntry, SEGMENT_FOOTER_TRAILER_LEN, SEGMENT_HEADER_LEN, SegmentFooter,
            SegmentHeader,
        },
    },
};

const FINGERPRINT_READ_CHUNK_LEN: usize = 64 * 1024;
const FINGERPRINT_HASH_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FINGERPRINT_HASH_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Clone, Copy)]
pub(super) struct SegmentOpenOptions {
    pub(super) expected_key_len: usize,
    pub(super) expected_value_layout: ValueLayout,
    pub(super) expected_block_checksum: BlockChecksumKind,
    pub(super) expected_value_payload_compression: ValuePayloadCompressionKind,
    pub(super) expected_fingerprint: SegmentFileFingerprint,
}

#[derive(Clone, Copy)]
pub(super) struct BlockReadOptions<'a> {
    pub(super) key_len: usize,
    pub(super) value_layout: ValueLayout,
    pub(super) block_checksum: BlockChecksumKind,
    pub(super) value_payload_compression: ValuePayloadCompressionKind,
    pub(super) verify_checksum: bool,
    pub(super) upper_key_bound: BlockKeyUpperBound<'a>,
}

/// Open segment handle with its sparse block index loaded into memory.
#[derive(Debug)]
pub(super) struct OpenedSegment {
    pub(super) file: File,
    pub(super) min_key: Vec<u8>,
    pub(super) max_key: Vec<u8>,
    pub(super) fingerprint: SegmentFileFingerprint,
    pub(super) block_index: Vec<BlockIndexEntry>,
}

impl OpenedSegment {
    /// Opens and validates one segment file.
    ///
    /// Returns `Ok(None)` when the file is missing, malformed, corrupt, or
    /// does not match the expected store geometry: a referenced segment that
    /// fails to open is miss space, not an error.
    pub(super) fn open(path: PathBuf, options: SegmentOpenOptions) -> Result<Option<Self>> {
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
            options.expected_key_len,
            options.expected_value_layout,
            options.expected_block_checksum,
            options.expected_value_payload_compression,
        ) {
            return Ok(None);
        }
        let footer = match read_footer(&file, options.expected_key_len) {
            Ok(footer) => footer,
            Err(error) if error.is_cache_miss_corruption() => return Ok(None),
            Err(error) => return Err(error),
        };
        let fingerprint = match segment_file_fingerprint(&file) {
            Ok(fingerprint) => fingerprint,
            Err(error) if error.is_cache_miss_corruption() => return Ok(None),
            Err(error) => return Err(error),
        };
        if fingerprint != options.expected_fingerprint {
            return Ok(None);
        }
        Ok(Some(Self {
            file,
            min_key: footer.min_key,
            max_key: footer.max_key,
            fingerprint,
            block_index: footer.block_index,
        }))
    }
}

pub(crate) fn segment_file_fingerprint(file: &File) -> Result<SegmentFileFingerprint> {
    use crate::format::CorruptionError;

    let len = file.metadata()?.len();
    let mut hash = FINGERPRINT_HASH_OFFSET;
    let mut offset = 0;
    let mut buffer = vec![0u8; FINGERPRINT_READ_CHUNK_LEN];
    while offset < len {
        let remaining = usize::try_from(len - offset).unwrap_or(usize::MAX);
        let read_len = remaining.min(buffer.len());
        let chunk = &mut buffer[..read_len];
        match read_exact_at(file, offset, chunk) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => {
                return Err(CorruptionError::SegmentFormat.into());
            }
            Err(error) => return Err(error.into()),
        }
        hash = fingerprint_hash_append(hash, chunk);
        offset += read_len as u64;
    }
    Ok(SegmentFileFingerprint { len, hash })
}

fn fingerprint_hash_append(mut hash: u64, bytes: &[u8]) -> u64 {
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FINGERPRINT_HASH_PRIME);
    }
    hash
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
pub(super) fn read_block(
    file: &File,
    entry: &BlockIndexEntry,
    options: BlockReadOptions<'_>,
) -> Result<DecodedBlock> {
    read_block_reusing(file, entry, options, Vec::new())
}

/// Reads and decodes one block while reusing a caller-owned backing buffer.
pub(super) fn read_block_reusing(
    file: &File,
    entry: &BlockIndexEntry,
    options: BlockReadOptions<'_>,
    mut bytes: Vec<u8>,
) -> Result<DecodedBlock> {
    let block_len = entry.block_len as usize;
    if bytes.len() < block_len {
        bytes.resize(block_len, 0);
    } else {
        bytes.truncate(block_len);
    }
    read_exact_at(file, entry.block_offset, &mut bytes)?;
    Ok(DecodedBlock::decode(bytes, entry, BlockDecodeOptions {
        key_len: options.key_len,
        value_layout: options.value_layout,
        block_checksum: options.block_checksum,
        value_payload_compression: options.value_payload_compression,
        verify_checksum: options.verify_checksum,
        upper_key_bound: options.upper_key_bound,
    })?)
}
