//! Low-level catalog filesystem primitives: staged file publication and the
//! advisory writer lock.
//!
//! Catalog files use one-file atomic publication. Segment commits stage and
//! sync every temporary file before renaming the complete batch and syncing
//! its parent directory once.

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
}

impl<'a> AtomicFilePublish<'a> {
    /// Creates an atomic publish operation.
    ///
    /// The temporary file is created in the same parent directory by adding a
    /// `.tmp` extension to `final_path`.
    pub(super) fn new(final_path: &'a Path) -> Option<Self> {
        let final_parent = final_path.parent()?;

        Some(Self {
            final_path,
            final_parent,
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
        let mut batch = StagedFileBatch::new(self.final_parent);
        let result = batch.stage_with(self.final_path.to_path_buf(), write)?;
        batch.publish()?;
        Ok(result)
    }
}

/// Owned set of synced temporary files awaiting one durability publication.
///
/// Staging never changes final paths. Publication consumes the batch, renames
/// every staged file, then syncs the shared parent directory once.
#[derive(Debug)]
pub(super) struct StagedFileBatch {
    parent: PathBuf,
    files: Vec<StagedFile>,
}

#[derive(Debug)]
struct StagedFile {
    temp_path: PathBuf,
    final_path: PathBuf,
}

impl StagedFileBatch {
    pub(super) fn new(parent: &Path) -> Self {
        Self {
            parent: parent.to_path_buf(),
            files: Vec::new(),
        }
    }

    /// Writes and syncs one temporary file without changing its final path.
    pub(super) fn stage_with<T>(
        &mut self,
        final_path: PathBuf,
        write: impl FnOnce(&mut File) -> Result<T>,
    ) -> Result<T> {
        debug_assert_eq!(final_path.parent(), Some(self.parent.as_path()));
        let temp_path = temp_path_for(&final_path);
        let mut file = File::create(&temp_path)?;
        let result = write(&mut file)?;
        file.sync_all()?;
        drop(file);
        self.files.push(StagedFile {
            temp_path,
            final_path,
        });
        Ok(result)
    }

    /// Makes every staged file durable under its final name.
    pub(super) fn publish(self) -> Result<()> {
        if self.files.is_empty() {
            return Ok(());
        }
        for file in self.files {
            fs::rename(file.temp_path, file.final_path)?;
        }
        sync_dir(&self.parent)
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

    #[test]
    fn staged_batch_keeps_all_final_paths_hidden_until_publication() {
        let tempdir = tempfile::tempdir().expect("tempdir should be created");
        let first = tempdir.path().join("first");
        let second = tempdir.path().join("second");
        let mut batch = StagedFileBatch::new(tempdir.path());

        batch
            .stage_with(first.clone(), |file| {
                file.write_all(b"one")?;
                Ok(())
            })
            .expect("first file should stage");
        batch
            .stage_with(second.clone(), |file| {
                file.write_all(b"two")?;
                Ok(())
            })
            .expect("second file should stage");

        assert!(!first.exists());
        assert!(!second.exists());
        assert!(temp_path_for(&first).exists());
        assert!(temp_path_for(&second).exists());

        batch.publish().expect("batch should publish");

        assert_eq!(fs::read(first).expect("first file should exist"), b"one");
        assert_eq!(fs::read(second).expect("second file should exist"), b"two");
    }

    #[cfg(unix)]
    #[test]
    fn sync_dir_accepts_directory_paths_on_unix() {
        let tempdir = tempfile::tempdir().expect("tempdir should be created");

        sync_dir(tempdir.path()).expect("directory fsync should succeed on Unix");
    }
}
