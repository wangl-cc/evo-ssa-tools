use std::{fs::File, io};

use crc32c::crc32c_append;

use crate::{
    binary::read_u32,
    error::{Error, Result},
    format::BlockIndexEntry,
    options::ValueLayout,
};

pub(crate) const BLOCK_HEADER_LEN: usize = 16;
const BLOCK_CHECKSUM_LEN: usize = 4;

#[derive(Debug)]
pub(crate) struct DecodedBlock {
    pub first_key: Vec<u8>,
    pub last_key: Vec<u8>,
    record_count: usize,
    key_len: usize,
    value_layout: ValueLayout,
    keys_offset: usize,
    value_offsets_offset: usize,
    value_lens_offset: usize,
    value_area_offset: usize,
    value_area_len: usize,
    bytes: Vec<u8>,
}

pub(crate) fn encode_block(
    entries: &[(Vec<u8>, Vec<u8>)],
    key_len: usize,
    value_layout: ValueLayout,
    target_block_size: usize,
) -> Vec<u8> {
    let record_count = entries.len();
    let keys_len = record_count * key_len;
    let value_area_len = entries.iter().map(|(_, value)| value.len()).sum::<usize>();
    let value_area_offset = match value_layout {
        ValueLayout::Variable => BLOCK_HEADER_LEN + keys_len + record_count * 8,
        ValueLayout::Fixed { .. } => BLOCK_HEADER_LEN + keys_len,
    };
    let logical_len = value_area_offset + value_area_len;
    let min_block_len = logical_len + BLOCK_CHECKSUM_LEN;
    let block_capacity = min_block_len.max(target_block_size);
    let mut block = Vec::with_capacity(block_capacity);
    block.extend_from_slice(
        &u32::try_from(record_count)
            .expect("record count should fit in u32")
            .to_le_bytes(),
    );
    block.extend_from_slice(&0u32.to_le_bytes());
    block.extend_from_slice(
        &u32::try_from(value_area_offset)
            .expect("value area offset should fit in u32")
            .to_le_bytes(),
    );
    block.extend_from_slice(
        &u32::try_from(value_area_len)
            .expect("value area length should fit in u32")
            .to_le_bytes(),
    );
    for (key, _) in entries {
        block.extend_from_slice(key);
    }
    if matches!(value_layout, ValueLayout::Variable) {
        let mut value_offset = 0usize;
        for (_, value) in entries {
            block.extend_from_slice(
                &u32::try_from(value_offset)
                    .expect("value offset should fit in u32")
                    .to_le_bytes(),
            );
            value_offset += value.len();
        }
        for (_, value) in entries {
            block.extend_from_slice(
                &u32::try_from(value.len())
                    .expect("value length should fit in u32")
                    .to_le_bytes(),
            );
        }
    }
    for (_, value) in entries {
        block.extend_from_slice(value);
    }
    let checksum_offset = logical_len;
    block.extend_from_slice(&0u32.to_le_bytes());
    if target_block_size > 0 && block.len() < target_block_size {
        block.resize(target_block_size, 0);
    }
    let total_len = u32::try_from(block.len()).expect("block length should fit in u32");
    block[4..8].copy_from_slice(&total_len.to_le_bytes());
    let checksum = block_crc(&block, checksum_offset);
    block[checksum_offset..checksum_offset + BLOCK_CHECKSUM_LEN]
        .copy_from_slice(&checksum.to_le_bytes());
    debug_assert!(block.len() >= value_area_offset + value_area_len + BLOCK_CHECKSUM_LEN);
    block
}

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

pub(crate) fn read_block_reusing(
    file: &File,
    entry: &BlockIndexEntry,
    key_len: usize,
    value_layout: ValueLayout,
    verify_checksum: bool,
    mut bytes: Vec<u8>,
) -> Result<DecodedBlock> {
    let block_len = usize::try_from(entry.block_len).expect("block len should fit");
    if bytes.len() < block_len {
        bytes.resize(block_len, 0);
    } else {
        bytes.truncate(block_len);
    }
    read_exact_at(file, entry.block_offset, &mut bytes)?;
    decode_block(bytes, entry, key_len, value_layout, verify_checksum)
}

#[cfg(unix)]
fn read_exact_at(file: &File, mut offset: u64, mut buffer: &mut [u8]) -> io::Result<()> {
    use std::os::unix::fs::FileExt;

    while !buffer.is_empty() {
        let read = file.read_at(buffer, offset)?;
        if read == 0 {
            return Err(io::ErrorKind::UnexpectedEof.into());
        }
        offset += u64::try_from(read).expect("read length should fit in u64");
        let (_, rest) = buffer.split_at_mut(read);
        buffer = rest;
    }
    Ok(())
}

#[cfg(not(unix))]
fn read_exact_at(file: &File, offset: u64, buffer: &mut [u8]) -> io::Result<()> {
    use std::io::{Read, Seek, SeekFrom};

    let mut file = file.try_clone()?;
    file.seek(SeekFrom::Start(offset))?;
    file.read_exact(buffer)
}

fn decode_block(
    bytes: Vec<u8>,
    entry: &BlockIndexEntry,
    key_len: usize,
    value_layout: ValueLayout,
    verify_checksum: bool,
) -> Result<DecodedBlock> {
    if bytes.len() < BLOCK_HEADER_LEN {
        return Err(Error::CorruptBlock);
    }
    let mut cursor = 0usize;
    let record_count =
        usize::try_from(read_u32(&bytes, &mut cursor)?).expect("record count should fit");
    let stored_len = usize::try_from(read_u32(&bytes, &mut cursor)?).expect("block len should fit");
    let value_area_offset =
        usize::try_from(read_u32(&bytes, &mut cursor)?).expect("value area offset should fit");
    let value_area_len =
        usize::try_from(read_u32(&bytes, &mut cursor)?).expect("value area len should fit");
    if stored_len > bytes.len()
        || record_count == 0
        || record_count != usize::try_from(entry.record_count).expect("record count should fit")
        || value_area_offset > stored_len
        || value_area_len > stored_len.saturating_sub(value_area_offset)
    {
        return Err(Error::CorruptBlock);
    }
    let logical_len = value_area_offset + value_area_len;
    let checksum_offset = logical_len;
    let checksum_end = checksum_offset
        .checked_add(BLOCK_CHECKSUM_LEN)
        .ok_or(Error::CorruptBlock)?;
    if checksum_end > stored_len {
        return Err(Error::CorruptBlock);
    }
    let stored_crc = u32::from_le_bytes(
        bytes[checksum_offset..checksum_end]
            .try_into()
            .expect("block crc slice"),
    );
    if verify_checksum && block_crc(&bytes, logical_len) != stored_crc {
        return Err(Error::CorruptBlock);
    }

    let keys_offset = BLOCK_HEADER_LEN;
    let keys_len = record_count
        .checked_mul(key_len)
        .ok_or(Error::CorruptBlock)?;
    let value_offsets_offset = keys_offset + keys_len;
    let value_lens_offset = match value_layout {
        ValueLayout::Variable => value_offsets_offset
            .checked_add(record_count * 4)
            .ok_or(Error::CorruptBlock)?,
        ValueLayout::Fixed { .. } => value_offsets_offset,
    };
    let expected_value_area_offset = match value_layout {
        ValueLayout::Variable => value_lens_offset
            .checked_add(record_count * 4)
            .ok_or(Error::CorruptBlock)?,
        ValueLayout::Fixed { value_len } => {
            let expected_value_len = record_count
                .checked_mul(value_len)
                .ok_or(Error::CorruptBlock)?;
            if value_area_len != expected_value_len {
                return Err(Error::CorruptBlock);
            }
            value_lens_offset
        }
    };
    if value_area_offset != expected_value_area_offset {
        return Err(Error::CorruptBlock);
    }

    let block = DecodedBlock {
        first_key: Vec::new(),
        last_key: Vec::new(),
        record_count,
        key_len,
        value_layout,
        keys_offset,
        value_offsets_offset,
        value_lens_offset,
        value_area_offset,
        value_area_len,
        bytes,
    };
    validate_value_table(&block)?;
    let first_key = block.key_at(0)?.to_vec();
    let last_key = block.key_at(record_count - 1)?.to_vec();
    Ok(DecodedBlock {
        first_key,
        last_key,
        ..block
    })
}

fn block_crc(bytes: &[u8], logical_len: usize) -> u32 {
    let logical_len = logical_len.min(bytes.len());
    crc32c_append(0, &bytes[..logical_len])
}

impl DecodedBlock {
    pub(crate) fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    pub(crate) fn find_value(&self, key: &[u8], key_len: usize) -> Option<Vec<u8>> {
        let index = self.partition_point_by_key(key, key_len);
        let record = self.record_at(index, key_len).ok()?;
        if record.key == key {
            Some(record.value.to_vec())
        } else {
            None
        }
    }

    pub(crate) fn records(&self, key_len: usize) -> impl Iterator<Item = ParsedRecord<'_>> {
        (0..self.record_count).filter_map(move |index| self.record_at(index, key_len).ok())
    }

    pub(crate) fn record_count(&self) -> usize {
        self.record_count
    }

    pub(crate) fn key_at_index(&self, index: usize) -> Result<&[u8]> {
        self.key_at(index)
    }

    pub(crate) fn record_at_index(&self, index: usize) -> Result<ParsedRecord<'_>> {
        self.record_at(index, self.key_len)
    }

    pub(crate) fn lower_bound_index(&self, key: &[u8]) -> usize {
        self.partition_point_by_key(key, self.key_len)
    }

    fn partition_point_by_key(&self, key: &[u8], key_len: usize) -> usize {
        let mut left = 0usize;
        let mut right = self.record_count;
        while left < right {
            let mid = left + (right - left) / 2;
            match self.key_at(mid) {
                Ok(candidate) if candidate < key => left = mid + 1,
                _ => right = mid,
            }
        }
        debug_assert!(key_len > 0);
        left
    }

    fn key_at(&self, index: usize) -> Result<&[u8]> {
        if index >= self.record_count {
            return Err(Error::CorruptBlock);
        }
        let start = self.keys_offset + index * self.key_len;
        let end = start + self.key_len;
        if end > self.value_offsets_offset {
            return Err(Error::CorruptBlock);
        }
        Ok(&self.bytes[start..end])
    }

    fn value_offset_at(&self, index: usize) -> Result<usize> {
        if let ValueLayout::Fixed { value_len } = self.value_layout {
            return index.checked_mul(value_len).ok_or(Error::CorruptBlock);
        }
        let start = self.value_offsets_offset + index * 4;
        let mut cursor = start;
        usize::try_from(read_u32(&self.bytes, &mut cursor)?).map_err(|_| Error::CorruptBlock)
    }

    fn value_len_at(&self, index: usize) -> Result<usize> {
        if let ValueLayout::Fixed { value_len } = self.value_layout {
            if index >= self.record_count {
                return Err(Error::CorruptBlock);
            }
            return Ok(value_len);
        }
        let start = self.value_lens_offset + index * 4;
        let mut cursor = start;
        usize::try_from(read_u32(&self.bytes, &mut cursor)?).map_err(|_| Error::CorruptBlock)
    }

    fn record_at(&self, index: usize, _key_len: usize) -> Result<ParsedRecord<'_>> {
        let key = self.key_at(index)?;
        let value_offset = self.value_offset_at(index)?;
        let value_len = self.value_len_at(index)?;
        if value_offset > self.value_area_len
            || value_len > self.value_area_len.saturating_sub(value_offset)
        {
            return Err(Error::CorruptBlock);
        }
        let start = self.value_area_offset + value_offset;
        let end = start + value_len;
        Ok(ParsedRecord {
            key,
            value: &self.bytes[start..end],
        })
    }
}

pub(crate) struct ParsedRecord<'a> {
    pub(crate) key: &'a [u8],
    pub(crate) value: &'a [u8],
}

fn validate_value_table(block: &DecodedBlock) -> Result<()> {
    if matches!(block.value_layout, ValueLayout::Fixed { .. }) {
        return Ok(());
    }
    let mut previous_end = 0usize;
    for index in 0..block.record_count {
        let value_offset = block.value_offset_at(index)?;
        let value_len = block.value_len_at(index)?;
        if value_offset < previous_end
            || value_offset > block.value_area_len
            || value_len > block.value_area_len.saturating_sub(value_offset)
        {
            return Err(Error::CorruptBlock);
        }
        previous_end = value_offset + value_len;
    }
    Ok(())
}
