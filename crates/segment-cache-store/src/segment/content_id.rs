//! Content identity derived from the complete bytes of one segment file.

use std::{
    fs::File,
    io::{self, Write},
};

use super::io::read_exact_at;
use crate::error::{CorruptionError, Result};

const DOMAIN_PREFIX: &[u8] = b"scs-segment-v1\0";
const READ_CHUNK_LEN: usize = 64 * 1024;
pub(crate) const SEGMENT_CONTENT_ID_LEN: usize = 32;

/// Stable BLAKE3 identity of one complete segment file.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct SegmentContentId([u8; SEGMENT_CONTENT_ID_LEN]);

impl SegmentContentId {
    pub(crate) fn from_bytes(bytes: [u8; SEGMENT_CONTENT_ID_LEN]) -> Self {
        Self(bytes)
    }

    pub(crate) fn as_bytes(&self) -> &[u8; SEGMENT_CONTENT_ID_LEN] {
        &self.0
    }

    pub(super) fn from_file(file: &File) -> Result<Self> {
        let len = file.metadata()?.len();
        let mut hasher = segment_hasher();
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
            hasher.update(chunk);
            offset += read_len as u64;
        }
        Ok(Self(*hasher.finalize().as_bytes()))
    }
}

/// Writer that identifies exactly the bytes accepted by its sink.
pub(super) struct ContentIdWriter<'a, W> {
    inner: &'a mut W,
    len: u64,
    hasher: blake3::Hasher,
}

impl<'a, W> ContentIdWriter<'a, W> {
    pub(super) fn new(inner: &'a mut W) -> Self {
        Self {
            inner,
            len: 0,
            hasher: segment_hasher(),
        }
    }

    pub(super) fn len(&self) -> u64 {
        self.len
    }

    pub(super) fn content_id(&self) -> SegmentContentId {
        SegmentContentId(*self.hasher.finalize().as_bytes())
    }
}

impl<W: Write> Write for ContentIdWriter<'_, W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let written = self.inner.write(bytes)?;
        let written_len = u64::try_from(written)
            .map_err(|_| io::Error::other("segment content length overflow"))?;
        self.len = self
            .len
            .checked_add(written_len)
            .ok_or_else(|| io::Error::other("segment content length overflow"))?;
        self.hasher.update(&bytes[..written]);
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

fn segment_hasher() -> blake3::Hasher {
    let mut hasher = blake3::Hasher::new();
    hasher.update(DOMAIN_PREFIX);
    hasher
}

#[cfg(test)]
mod tests {
    use std::io::{Seek, Write};

    use super::*;

    #[test]
    fn write_time_content_id_matches_complete_file_scan() {
        let tempdir = tempfile::tempdir().expect("tempdir should be created");
        let path = tempdir.path().join("segment");
        let mut file = File::options()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .expect("segment file should be created");
        let write_time_identity = {
            let mut writer = ContentIdWriter::new(&mut file);
            writer
                .write_all(b"header")
                .expect("first write should succeed");
            writer
                .write_all(&[0, 1, 2, 3, 4])
                .expect("second write should succeed");
            (writer.len(), writer.content_id())
        };
        file.flush().expect("segment file should flush");
        file.rewind().expect("segment file should rewind");

        assert_eq!(
            write_time_identity.0,
            file.metadata().expect("metadata").len()
        );
        assert_eq!(
            write_time_identity.1,
            SegmentContentId::from_file(&file).expect("file scan should identify segment")
        );
    }

    #[test]
    fn content_id_uses_the_segment_domain_prefix() {
        let mut expected = blake3::Hasher::new();
        expected.update(DOMAIN_PREFIX);
        expected.update(b"segment bytes");

        let mut sink = Vec::new();
        let actual = {
            let mut writer = ContentIdWriter::new(&mut sink);
            writer.write_all(b"segment bytes").expect("write succeeds");
            writer.content_id()
        };

        assert_eq!(actual.as_bytes(), expected.finalize().as_bytes());
    }

    #[test]
    fn same_length_value_changes_produce_different_content_ids() {
        let mut first = Vec::new();
        let first_id = {
            let mut writer = ContentIdWriter::new(&mut first);
            writer.write_all(b"key=value-a").expect("write succeeds");
            writer.content_id()
        };
        let mut second = Vec::new();
        let second_id = {
            let mut writer = ContentIdWriter::new(&mut second);
            writer.write_all(b"key=value-b").expect("write succeeds");
            writer.content_id()
        };

        assert_eq!(first.len(), second.len());
        assert_ne!(first_id, second_id);
    }
}
