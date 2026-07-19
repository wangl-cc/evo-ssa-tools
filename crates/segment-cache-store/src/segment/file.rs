//! File-level access to immutable segment files: open-and-validate, block
//! reads.
//!
//! Block and segment codecs own byte interpretation; this module reads the
//! required ranges and applies the open-time validation policy (a segment that
//! fails any check is treated as absent, never as an error).

use std::{fs::File, path::PathBuf};

use super::{
    Segment, SegmentContentId, SegmentGeometry,
    format::{SEGMENT_FOOTER_TRAILER_LEN, SEGMENT_HEADER_LEN, SegmentFooter, SegmentHeader},
    index::SegmentIndex,
    io::read_exact_at,
};
#[cfg(feature = "value-compression")]
use crate::block::ValuePayloadDecoder;
use crate::{
    binary::BinaryCursor,
    block::{BlockDecodeOptions, BlockKeyRangeRef, DecodedBlock},
    error::{CorruptionError, Result},
    limits::{MAX_ENCODED_BLOCK_LEN, MAX_FOOTER_LEN},
};

#[derive(Clone, Copy)]
pub(crate) struct SegmentOpenOptions<'a> {
    pub(crate) geometry: SegmentGeometry,
    pub(crate) expected_segment_len: u64,
    pub(crate) expected_content_id: SegmentContentId,
    pub(crate) verify_content_id: bool,
    pub(crate) expected_min_key: &'a [u8],
    pub(crate) expected_max_key: &'a [u8],
}

#[derive(Clone, Copy)]
pub(crate) struct BlockReadOptions {
    pub(crate) geometry: SegmentGeometry,
    #[cfg(feature = "block-checksum")]
    pub(crate) verify_lookup_checksum: bool,
}

/// Open segment handle with its sparse block index loaded into memory.
#[derive(Debug)]
pub(super) struct OpenedSegment {
    pub(crate) file: File,
    pub(crate) min_key: Vec<u8>,
    pub(crate) max_key: Vec<u8>,
    pub(crate) block_index: SegmentIndex,
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
        if file.metadata()?.len() != options.expected_segment_len {
            return Ok(None);
        }
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
        if options.verify_content_id
            && match Self::verify_file(
                &file,
                &footer.block_index,
                options.geometry,
                options.expected_content_id,
            ) {
                Ok(verified) => !verified,
                Err(error) if error.is_cache_miss_corruption() => return Ok(None),
                Err(error) => return Err(error),
            }
        {
            return Ok(None);
        }
        Ok(Some(Self {
            file,
            min_key: options.expected_min_key.to_vec(),
            max_key: options.expected_max_key.to_vec(),
            block_index: footer.block_index,
        }))
    }

    /// Verifies one externally materialized segment before it is trusted.
    fn verify_file(
        file: &File,
        index: &SegmentIndex,
        geometry: SegmentGeometry,
        expected_content_id: SegmentContentId,
    ) -> Result<bool> {
        if SegmentContentId::from_file(file)? != expected_content_id {
            return Ok(false);
        }
        verify_blocks(file, index, geometry)?;
        Ok(true)
    }
}

fn verify_blocks(file: &File, index: &SegmentIndex, geometry: SegmentGeometry) -> Result<()> {
    #[cfg(feature = "value-compression")]
    let mut payload_decoder = ValuePayloadDecoder::new(geometry.value_payload_compression);
    for block_index in 0..index.len() {
        let block = read_block(file, index, block_index, BlockReadOptions {
            geometry,
            #[cfg(feature = "block-checksum")]
            verify_lookup_checksum: true,
        })?;
        #[cfg(feature = "block-checksum")]
        block.verify_payload_checksum()?;
        #[cfg(feature = "value-compression")]
        let _decoded_payload = block.decode_payload_if_needed(&mut payload_decoder)?;
        #[cfg(not(feature = "value-compression"))]
        let _payload = block.payload_bytes()?;
    }
    Ok(())
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
    if footer_len > MAX_FOOTER_LEN as u64 {
        return Err(CorruptionError::SegmentFormat.into());
    }
    if file_len < footer_len {
        return Err(CorruptionError::SegmentFormat.into());
    }
    let data_end = file_len - footer_len;
    let footer_len = usize::try_from(footer_len).map_err(|_| CorruptionError::SegmentFormat)?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(footer_len)
        .map_err(|_| CorruptionError::SegmentFormat)?;
    bytes.resize(footer_len, 0);
    read_exact_at(file, data_end, &mut bytes)?;
    Ok(SegmentFooter::from_bytes(
        bytes,
        key_len,
        data_end,
        expected_min_key,
        expected_max_key,
    )?)
}

/// Reads and decodes one block, allocating a fresh buffer.
pub(super) fn read_block(
    file: &File,
    index: &SegmentIndex,
    block_index: usize,
    options: BlockReadOptions,
) -> Result<DecodedBlock> {
    read_block_reusing(file, index, block_index, options, Vec::new())
}

/// Reads and decodes one block while reusing a caller-owned backing buffer.
pub(super) fn read_block_reusing(
    file: &File,
    index: &SegmentIndex,
    block_index: usize,
    options: BlockReadOptions,
    mut bytes: Vec<u8>,
) -> Result<DecodedBlock> {
    let entry = index.entry(block_index);
    let block_len = usize::try_from(entry.byte_range.end - entry.byte_range.start)
        .map_err(|_| CorruptionError::SegmentFormat)?;
    if block_len > MAX_ENCODED_BLOCK_LEN {
        return Err(CorruptionError::SegmentFormat.into());
    }
    if bytes.capacity() < block_len {
        bytes
            .try_reserve_exact(block_len - bytes.len())
            .map_err(|_| CorruptionError::SegmentFormat)?;
    }
    if bytes.len() < block_len {
        bytes.resize(block_len, 0);
    } else {
        bytes.truncate(block_len);
    }
    read_exact_at(file, entry.byte_range.start, &mut bytes)?;
    let expected = index.key_range(block_index);
    Ok(DecodedBlock::decode(bytes, BlockDecodeOptions {
        expected_key_range: BlockKeyRangeRef {
            segment_prefix: index.segment_prefix().as_slice(),
            extra_prefix: expected.extra_prefix,
            min_suffix: expected.min_suffix,
            max_suffix: expected.max_suffix,
        },
        key_len: options.geometry.key_len,
        value_layout: options.geometry.value_layout,
        block_checksum: options.geometry.block_checksum,
        value_payload_compression: options.geometry.value_payload_compression,
        #[cfg(feature = "block-checksum")]
        verify_lookup_checksum: options.verify_lookup_checksum,
    })?)
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Seek, SeekFrom, Write};

    use super::*;
    #[cfg(feature = "value-compression")]
    use crate::block::ValuePayloadCompressionPolicy;
    use crate::{
        block::{BlockChecksumKind, ValuePayloadCompressionKind},
        record::{EntryRef, EntrySource},
        segment::SegmentWriter,
        value::ValueLayout,
    };

    struct OneEntry;

    impl EntrySource for OneEntry {
        fn len(&self) -> usize {
            1
        }

        fn entry(&self, index: usize) -> EntryRef<'_> {
            assert_eq!(index, 0);
            EntryRef::new(b"key1", b"value")
        }
    }

    #[test]
    fn explicit_single_segment_verification_detects_same_length_corruption() -> Result<()> {
        let tempdir = tempfile::tempdir()?;
        let path = tempdir.path().join("segment.seg");
        let geometry = SegmentGeometry {
            key_len: 4,
            value_layout: ValueLayout::VARIABLE,
            block_checksum: BlockChecksumKind::None,
            value_payload_compression: ValuePayloadCompressionKind::None,
        };
        let mut file = File::options()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)?;
        let metadata = segment_writer(geometry).write(&mut file, &OneEntry)?;
        file.sync_all()?;
        drop(file);
        let options = SegmentOpenOptions {
            geometry,
            expected_segment_len: metadata.segment_len(),
            expected_content_id: metadata.content_id(),
            verify_content_id: true,
            expected_min_key: &metadata.min_key,
            expected_max_key: &metadata.max_key,
        };

        assert!(OpenedSegment::open(path.clone(), options)?.is_some());

        let payload_byte = metadata.footer.block_index.entry(0).byte_range.end - 1;
        let mut file = File::options().read(true).write(true).open(&path)?;
        file.seek(SeekFrom::Start(payload_byte))?;
        let mut byte = [0];
        file.read_exact(&mut byte)?;
        byte[0] ^= 0xff;
        file.seek(SeekFrom::Start(payload_byte))?;
        file.write_all(&byte)?;
        file.sync_all()?;

        assert!(OpenedSegment::open(path, options)?.is_none());
        Ok(())
    }

    #[cfg(feature = "value-compression")]
    fn segment_writer(geometry: SegmentGeometry) -> SegmentWriter {
        SegmentWriter::new(geometry, ValuePayloadCompressionPolicy::DEFAULT, 16 * 1024)
    }

    #[cfg(not(feature = "value-compression"))]
    fn segment_writer(geometry: SegmentGeometry) -> SegmentWriter {
        SegmentWriter::new(geometry, 16 * 1024)
    }
}
