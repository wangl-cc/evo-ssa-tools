use crate::error::{Error, Result};

pub(crate) fn read_u32(bytes: &[u8], cursor: &mut usize) -> Result<u32> {
    if *cursor + 4 > bytes.len() {
        return Err(Error::UnsupportedFormatVersion { version: 0 });
    }
    let value = u32::from_le_bytes(bytes[*cursor..*cursor + 4].try_into().expect("u32 slice"));
    *cursor += 4;
    Ok(value)
}

pub(crate) fn read_u64(bytes: &[u8], cursor: &mut usize) -> Result<u64> {
    if *cursor + 8 > bytes.len() {
        return Err(Error::UnsupportedFormatVersion { version: 0 });
    }
    let value = u64::from_le_bytes(bytes[*cursor..*cursor + 8].try_into().expect("u64 slice"));
    *cursor += 8;
    Ok(value)
}

pub(crate) fn read_bytes(bytes: &[u8], cursor: &mut usize, len: usize) -> Result<Vec<u8>> {
    if *cursor + len > bytes.len() {
        return Err(Error::UnsupportedFormatVersion { version: 0 });
    }
    let value = bytes[*cursor..*cursor + len].to_vec();
    *cursor += len;
    Ok(value)
}
