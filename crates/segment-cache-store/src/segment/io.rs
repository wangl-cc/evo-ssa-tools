//! Positioned reads shared by immutable segment file operations.

use std::fs::File;

pub(super) fn read_exact_at(
    file: &File,
    mut offset: u64,
    mut buffer: &mut [u8],
) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;

    while !buffer.is_empty() {
        let read = file.read_at(buffer, offset)?;
        if read == 0 {
            return Err(std::io::ErrorKind::UnexpectedEof.into());
        }
        offset += read as u64;
        let (_, rest) = buffer.split_at_mut(read);
        buffer = rest;
    }
    Ok(())
}
