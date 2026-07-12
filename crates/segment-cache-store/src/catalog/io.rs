//! Low-level catalog filesystem primitives: atomic publication and the
//! advisory writer lock.
//!
//! Store catalog files and segment files share the same publication protocol:
//! write a temporary file, fsync it, rename it into place, then fsync the
//! final parent directory. The encoded content may be a byte slice or a
//! streaming writer; this module owns the filesystem protocol for both cases.

use std::{
    fs::{self, File, TryLockError},
    io::Write,
    path::{Path, PathBuf},
};

use crate::error::{InputError, Result};

/// Prepared atomic publish operation for one file.
#[derive(Clone, Debug)]
pub(crate) struct AtomicFilePublish<'a> {
    final_path: &'a Path,
    final_parent: &'a Path,
    temp_path: PathBuf,
}

impl<'a> AtomicFilePublish<'a> {
    /// Creates an atomic publish operation.
    ///
    /// The temporary file is created in the same parent directory by adding a
    /// `.tmp` extension to `final_path`.
    pub(super) fn new(final_path: &'a Path) -> Option<Self> {
        let final_parent = final_path.parent()?;
        let temp_path = temp_path_for(final_path);

        Some(Self {
            final_path,
            final_parent,
            temp_path,
        })
    }

    /// Writes bytes to the temporary file and publishes it atomically.
    pub(super) fn write_bytes(&self, bytes: &[u8]) -> Result<()> {
        self.write_with(|file| {
            file.write_all(bytes)?;
            Ok(())
        })
    }

    /// Streams content into the temporary file and publishes it atomically.
    pub(crate) fn write_with<T>(&self, write: impl FnOnce(&mut File) -> Result<T>) -> Result<T> {
        let mut file = File::create(&self.temp_path)?;
        let result = write(&mut file)?;
        file.sync_all()?;
        drop(file);
        fs::rename(&self.temp_path, self.final_path)?;
        sync_dir(self.final_parent)?;
        Ok(result)
    }
}

/// Returns the temporary path used while publishing `final_path`.
pub(super) fn temp_path_for(final_path: &Path) -> PathBuf {
    final_path.with_added_extension("tmp")
}

fn sync_dir(path: &Path) -> Result<()> {
    let dir = File::open(path)?;
    dir.sync_all()?;
    Ok(())
}

/// Held advisory single-writer lock for one store root.
///
/// A writer (commit, GC, sync ingestion) holds this lock for the lifetime of
/// its `Store` handle so a second writer fails fast instead of corrupting
/// visibility. Read-only opens do not take the lock. The lock is advisory: it
/// only coordinates processes that cooperate by acquiring it. It is released
/// when this value is dropped, which closes the underlying file descriptor.
#[derive(Debug)]
pub(crate) struct WriterLock {
    // The lock is bound to the open file description, so this field exists to
    // keep the lock held for the lifetime of `WriterLock`.
    _file: File,
}

impl WriterLock {
    /// Acquires an exclusive non-blocking advisory lock on `path`.
    ///
    /// Returns [`InputError::WriterLocked`] when another writer already holds
    /// the lock. The lock file itself is stable and is never atomically
    /// replaced; this matters because advisory locks attach to the opened file.
    pub(super) fn acquire(path: &Path) -> Result<Self> {
        let file = File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
        match file.try_lock() {
            Ok(()) => Ok(Self { _file: file }),
            Err(TryLockError::WouldBlock) => Err(InputError::WriterLocked.into()),
            Err(TryLockError::Error(error)) => Err(error.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, io::Write};

    use super::*;
    use crate::Error;

    #[test]
    fn write_bytes_publishes_final_file_and_removes_temp_file() {
        let tempdir = tempfile::tempdir().expect("tempdir should be created");
        let final_path = tempdir.path().join("file");
        let temp_path = tempdir.path().join("file.tmp");
        let publish = AtomicFilePublish::new(&final_path).expect("publish path should be valid");

        publish
            .write_bytes(b"published")
            .expect("publish should succeed");

        assert_eq!(
            fs::read(&final_path).expect("final file should be readable"),
            b"published"
        );
        assert!(
            !temp_path.exists(),
            "rename should consume the temporary file"
        );
    }

    #[test]
    fn write_with_streams_content_and_returns_writer_result() {
        let tempdir = tempfile::tempdir().expect("tempdir should be created");
        let final_path = tempdir.path().join("file");
        let publish = AtomicFilePublish::new(&final_path).expect("publish path should be valid");

        let byte_count = publish
            .write_with(|file| {
                file.write_all(b"streamed")?;
                Ok(8usize)
            })
            .expect("publish should succeed");

        assert_eq!(byte_count, 8);
        assert_eq!(
            fs::read(&final_path).expect("final file should be readable"),
            b"streamed"
        );
    }

    #[test]
    fn writer_error_does_not_replace_existing_final_file() {
        let tempdir = tempfile::tempdir().expect("tempdir should be created");
        let final_path = tempdir.path().join("file");
        fs::write(&final_path, b"old").expect("existing final file should be written");
        let publish = AtomicFilePublish::new(&final_path).expect("publish path should be valid");

        let error = publish
            .write_with(|file| {
                file.write_all(b"partial")?;
                Err::<(), _>(Error::Io(std::io::Error::other("intentional failure")))
            })
            .expect_err("writer failure should propagate");

        assert!(matches!(error, Error::Io(_)));
        assert_eq!(
            fs::read(&final_path).expect("final file should still be readable"),
            b"old",
            "failed temp writes must not replace the visible file"
        );
    }

    #[cfg(unix)]
    #[test]
    fn sync_dir_accepts_directory_paths_on_unix() {
        let tempdir = tempfile::tempdir().expect("tempdir should be created");

        sync_dir(tempdir.path()).expect("directory fsync should succeed on Unix");
    }
}
