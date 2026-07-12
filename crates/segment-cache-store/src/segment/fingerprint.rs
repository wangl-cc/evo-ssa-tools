//! Stable identity derived from the complete bytes of one segment file.

use std::{
    fs::File,
    io::{self, Write},
};

use super::io::read_exact_at;
use crate::error::{CorruptionError, Result};

const READ_CHUNK_LEN: usize = 64 * 1024;
const HASH_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const HASH_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SegmentFingerprint {
    pub(crate) len: u64,
    pub(crate) hash: u64,
}

impl SegmentFingerprint {
    pub(super) fn from_file(file: &File) -> Result<Self> {
        let len = file.metadata()?.len();
        let mut hash = HASH_OFFSET;
        let mut offset = 0;
        let mut buffer = vec![0u8; READ_CHUNK_LEN];
        while offset < len {
            let remaining = usize::try_from(len - offset).unwrap_or(usize::MAX);
            let read_len = remaining.min(buffer.len());
            let chunk = &mut buffer[..read_len];
            match read_exact_at(file, offset, chunk) {
                Ok(()) => {}
                Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => {
                    return Err(CorruptionError::SegmentFormat.into());
                }
                Err(error) => return Err(error.into()),
            }
            hash = append_hash(hash, chunk);
            offset += read_len as u64;
        }
        Ok(Self { len, hash })
    }
}

/// Writer that fingerprints exactly the bytes accepted by its sink.
pub(super) struct FingerprintWriter<'a, W> {
    inner: &'a mut W,
    len: u64,
    hash: u64,
}

impl<'a, W> FingerprintWriter<'a, W> {
    pub(super) fn new(inner: &'a mut W) -> Self {
        Self {
            inner,
            len: 0,
            hash: HASH_OFFSET,
        }
    }

    pub(super) fn fingerprint(&self) -> SegmentFingerprint {
        SegmentFingerprint {
            len: self.len,
            hash: self.hash,
        }
    }
}

impl<W: Write> Write for FingerprintWriter<'_, W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let written = self.inner.write(bytes)?;
        let written_len = u64::try_from(written)
            .map_err(|_| io::Error::other("segment fingerprint length overflow"))?;
        self.len = self
            .len
            .checked_add(written_len)
            .ok_or_else(|| io::Error::other("segment fingerprint length overflow"))?;
        self.hash = append_hash(self.hash, &bytes[..written]);
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

fn append_hash(mut hash: u64, bytes: &[u8]) -> u64 {
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(HASH_PRIME);
    }
    hash
}

#[cfg(test)]
mod tests {
    use std::io::{Seek, Write};

    use super::*;

    #[test]
    fn write_time_fingerprint_matches_file_scan() {
        let tempdir = tempfile::tempdir().expect("tempdir should be created");
        let path = tempdir.path().join("segment");
        let mut file = File::options()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .expect("segment file should be created");
        let write_time_fingerprint = {
            let mut writer = FingerprintWriter::new(&mut file);
            writer
                .write_all(b"header")
                .expect("first write should succeed");
            writer
                .write_all(&[0, 1, 2, 3, 4])
                .expect("second write should succeed");
            writer.fingerprint()
        };
        file.flush().expect("segment file should flush");
        file.rewind().expect("segment file should rewind");

        assert_eq!(
            write_time_fingerprint,
            SegmentFingerprint::from_file(&file).expect("file scan should fingerprint segment")
        );
    }
}
