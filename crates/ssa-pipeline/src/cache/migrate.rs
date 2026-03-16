//! Store migration utilities.
//!
//! Use these helpers when you need to move cached data between storage backends or transcode it
//! from one codec to another.
//!
//! # Choosing the right function
//!
//! - [`copy`]: same codec, new storage backend (e.g., `Fjall3Store` → new keyspace).
//! - [`copy_transcoded`]: new codec, same or new storage backend (e.g., `Bitcode06` → `Postcard`).

use super::{codec::CodecEngine, storage};
use crate::error::Result;

/// Stores that support full-scan iteration over all cached entries.
///
/// This trait is the foundation for store migration.
pub trait IterableStore {
    /// Call `f` for every `(key, encoded_value)` pair in the store.
    ///
    /// Iteration stops early if `f` returns `Err`.
    fn iter_encoded<F>(&self, f: F) -> Result<()>
    where
        F: FnMut(&[u8], &[u8]) -> Result<()>;
}

impl<H> IterableStore for storage::HashMapStore<H>
where
    H: std::hash::BuildHasher + Send + Sync,
{
    fn iter_encoded<F>(&self, mut f: F) -> Result<()>
    where
        F: FnMut(&[u8], &[u8]) -> Result<()>,
    {
        let map = self.inner.read();
        for (key, value) in map.iter() {
            f(key, value)?;
        }
        Ok(())
    }
}

#[cfg(feature = "fjall2")]
impl IterableStore for storage::Fjall2Store {
    fn iter_encoded<F>(&self, mut f: F) -> Result<()>
    where
        F: FnMut(&[u8], &[u8]) -> Result<()>,
    {
        for result in self.handle.iter() {
            let (key, value) = result.map_err(storage::StorageError::from)?;
            f(key.as_ref(), value.as_ref())?;
        }
        Ok(())
    }
}

#[cfg(feature = "fjall3")]
impl IterableStore for super::storage::Fjall3Store {
    fn iter_encoded<F>(&self, mut f: F) -> Result<()>
    where
        F: FnMut(&[u8], &[u8]) -> Result<()>,
    {
        for guard in self.handle.iter() {
            let (key, value) = guard.into_inner().map_err(storage::StorageError::from)?;
            f(key.as_ref(), value.as_ref())?;
        }
        Ok(())
    }
}

#[cfg(feature = "redb")]
impl IterableStore for super::storage::RedbStore {
    fn iter_encoded<F>(&self, mut f: F) -> Result<()>
    where
        F: FnMut(&[u8], &[u8]) -> Result<()>,
    {
        use redb::{ReadableDatabase, ReadableTable, TableDefinition};
        type BytesTable<'n> = TableDefinition<'n, &'static [u8], &'static [u8]>;

        let tx = self.db.begin_read().map_err(storage::StorageError::from)?;
        let definition = BytesTable::new(self.table_name.as_ref());
        let table = tx
            .open_table(definition)
            .map_err(storage::StorageError::from)?;
        for entry in table.iter().map_err(storage::StorageError::from)? {
            let (key, value) = entry.map_err(storage::StorageError::from)?;
            f(key.value(), value.value())?;
        }
        Ok(())
    }
}

/// Statistics returned by [`copy`] and [`copy_transcoded`].
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct MigrateStats {
    /// Number of entries successfully written to the destination store.
    pub migrated: usize,
    /// Number of entries skipped because re-encoding was rejected by the destination engine.
    ///
    /// Always zero for [`copy`] (raw bytes are never re-encoded).
    pub skipped: usize,
}

/// Copy all raw encoded entries from `src` to `dst` without any transcoding.
///
/// Use this when migrating to a new storage backend while keeping the same codec engine.
/// Entries already present in `dst` are overwritten.
///
/// # Undefined Behavior
///
/// `src` and `dst` must be distinct stores. Passing the same store for both is unsound:
/// some backends (e.g. [`HashMapStore`](super::storage::HashMapStore)) hold a read lock for the
/// entire iteration, so any write attempted by the closure will deadlock.
///
/// # Example
///
/// ```rust
/// use ssa_pipeline::{cache::migrate::copy, prelude::*};
///
/// let src = DefaultHashMapStore::default();
/// let mut engine = Bitcode06::default();
/// src.store(b"k1", &mut engine, &1u32).unwrap();
/// src.store(b"k2", &mut engine, &2u32).unwrap();
///
/// // In practice, `dst` would be a different storage backend (e.g. a database).
/// let dst = DefaultHashMapStore::default();
/// let stats = copy(&src, &dst).unwrap();
///
/// assert_eq!(stats.migrated, 2);
/// assert_eq!(stats.skipped, 0);
/// assert_eq!(dst.fetch::<u32, _>(b"k1", &mut engine).unwrap(), Some(1));
/// assert_eq!(dst.fetch::<u32, _>(b"k2", &mut engine).unwrap(), Some(2));
/// ```
pub fn copy<Src, Dst>(src: &Src, dst: &Dst) -> Result<MigrateStats>
where
    Src: IterableStore,
    Dst: storage::CacheStore,
{
    let mut stats = MigrateStats::default();
    src.iter_encoded(|key, encoded| {
        dst.store_encoded(key, encoded)?;
        stats.migrated += 1;
        Ok(())
    })?;
    Ok(stats)
}

/// Copy all entries from `src` to `dst`, decoding with `from_engine` and re-encoding with
/// `to_engine`.
///
/// This is the primary migration path when changing codec engines (e.g., `Bitcode06` →
/// `Postcard`). Entries that fail to re-encode are skipped with a warning, matching the
/// behavior of [`CacheStore::store`](storage::CacheStore::store). Entries that fail to decode
/// abort the migration with an error.
///
/// # Panics / deadlocks
///
/// `src` and `dst` must be distinct stores. Passing the same store for both is unsound:
/// some backends (e.g. [`HashMapStore`](super::storage::HashMapStore)) hold a read lock for the
/// entire iteration, so any write attempted by the closure will deadlock.
///
/// In-place transcoding is intentionally not provided: a failure partway through would leave
/// the store in a mixed-codec state that cannot be rolled back.
///
/// # Example
///
/// ```
/// use ssa_pipeline::{cache::migrate::copy_transcoded, prelude::*};
///
/// let src = DefaultHashMapStore::default();
/// let mut from_engine = Bitcode06::default();
/// src.store(b"key", &mut from_engine, &100u64).unwrap();
///
/// let dst = DefaultHashMapStore::default();
/// // Add crc32c checksums to the encoded values
/// let mut to_engine = CheckedCodec::new(Bitcode06::default());
/// let stats =
///     copy_transcoded::<u64, _, _, _, _>(&src, &mut from_engine, &dst, &mut to_engine).unwrap();
///
/// assert_eq!(stats.migrated, 1);
/// assert_eq!(stats.skipped, 0);
/// assert_eq!(
///     dst.fetch::<u64, _>(b"key", &mut to_engine).unwrap(),
///     Some(100)
/// );
/// ```
pub fn copy_transcoded<T, Src, Dst, FromCE, ToCE>(
    src: &Src,
    from_engine: &mut FromCE,
    dst: &Dst,
    to_engine: &mut ToCE,
) -> Result<MigrateStats>
where
    Src: IterableStore,
    Dst: storage::CacheStore,
    FromCE: CodecEngine<T>,
    ToCE: CodecEngine<T>,
{
    let mut stats = MigrateStats::default();
    src.iter_encoded(|key, encoded| {
        let value = from_engine.decode(encoded)?;
        match to_engine.encode(&value) {
            Ok(re_encoded) => {
                dst.store_encoded(key, re_encoded)?;
                stats.migrated += 1;
            }
            Err(reason) => {
                eprintln!("[ssa-pipeline] skipping migration entry: {reason}");
                stats.skipped += 1;
            }
        }
        Ok(())
    })?;
    Ok(stats)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::cache::{
        codec::{CodecEngine, Error as CodecError, SkipReason, fixtures::FixtureEngine},
        storage::{CacheStore, DefaultHashMapStore},
    };

    fn make_store(entries: &[(&[u8], u32)]) -> DefaultHashMapStore {
        let store = DefaultHashMapStore::default();
        let mut engine = FixtureEngine::default();
        for &(key, value) in entries {
            store.store::<u32, _>(key, &mut engine, &value).unwrap();
        }
        store
    }

    // -- copy ---------------------------------------------------------------

    #[test]
    fn copy_copies_all_entries() -> Result<()> {
        let src = make_store(&[(b"k1", 1), (b"k2", 2), (b"k3", 3)]);
        let dst = DefaultHashMapStore::default();
        let stats = copy(&src, &dst)?;
        assert_eq!(stats.migrated, 3);
        assert_eq!(stats.skipped, 0);

        let mut engine = FixtureEngine::default();
        assert_eq!(dst.fetch::<u32, _>(b"k1", &mut engine)?, Some(1));
        assert_eq!(dst.fetch::<u32, _>(b"k2", &mut engine)?, Some(2));
        assert_eq!(dst.fetch::<u32, _>(b"k3", &mut engine)?, Some(3));
        Ok(())
    }

    #[test]
    fn copy_empty_source() -> Result<()> {
        let src = DefaultHashMapStore::default();
        let dst = DefaultHashMapStore::default();
        let stats = copy(&src, &dst)?;
        assert_eq!(stats, MigrateStats::default());
        Ok(())
    }

    // -- copy_transcoded ----------------------------------------------------

    #[test]
    fn copy_transcoded_roundtrips() -> Result<()> {
        let src = make_store(&[(b"k1", 1), (b"k2", 2), (b"k3", 3)]);
        let dst = DefaultHashMapStore::default();
        let stats = copy_transcoded::<u32, _, _, _, _>(
            &src,
            &mut FixtureEngine::default(),
            &dst,
            &mut FixtureEngine::default(),
        )?;
        assert_eq!(stats.migrated, 3);
        assert_eq!(stats.skipped, 0);

        let mut engine = FixtureEngine::default();
        assert_eq!(dst.fetch::<u32, _>(b"k1", &mut engine)?, Some(1));
        assert_eq!(dst.fetch::<u32, _>(b"k2", &mut engine)?, Some(2));
        assert_eq!(dst.fetch::<u32, _>(b"k3", &mut engine)?, Some(3));
        Ok(())
    }

    #[test]
    fn copy_transcoded_aborts_on_decode_error() {
        let src = DefaultHashMapStore::default();
        src.store_encoded(b"bad", b"not valid fixture encoding")
            .unwrap();
        let dst = DefaultHashMapStore::default();
        let result = copy_transcoded::<u32, _, _, _, _>(
            &src,
            &mut FixtureEngine::default(),
            &dst,
            &mut FixtureEngine::default(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn copy_transcoded_skips_on_encode_rejection() -> Result<()> {
        #[derive(Default)]
        struct AlwaysSkipEngine;
        impl CodecEngine<u32> for AlwaysSkipEngine {
            fn encode(&mut self, _: &u32) -> std::result::Result<&[u8], SkipReason> {
                Err(SkipReason::EncodedValueTooLarge {
                    encoded_len: 100,
                    max_len: 0,
                })
            }

            fn decode(&mut self, _: &[u8]) -> std::result::Result<u32, CodecError> {
                unreachable!()
            }
        }

        let src = make_store(&[(b"k1", 1), (b"k2", 2), (b"k3", 3)]);
        let dst = DefaultHashMapStore::default();
        let stats = copy_transcoded::<u32, _, _, _, _>(
            &src,
            &mut FixtureEngine::default(),
            &dst,
            &mut AlwaysSkipEngine,
        )?;
        assert_eq!(stats.migrated, 0);
        assert_eq!(stats.skipped, 3);
        Ok(())
    }

    #[test]
    fn copy_transcoded_empty_source() -> Result<()> {
        let src = DefaultHashMapStore::default();
        let dst = DefaultHashMapStore::default();
        let stats = copy_transcoded::<u32, _, _, _, _>(
            &src,
            &mut FixtureEngine::default(),
            &dst,
            &mut FixtureEngine::default(),
        )?;
        assert_eq!(stats, MigrateStats::default());
        Ok(())
    }

    // -- IterableStore: redb ------------------------------------------------

    #[cfg(feature = "redb")]
    #[test]
    fn redb_iter_encoded_roundtrips() -> Result<()> {
        use std::sync::Arc;

        use storage::{RedbStore, StorageError};

        let file = tempfile::NamedTempFile::new().unwrap();
        let db = Arc::new(redb::Database::create(file.path()).map_err(StorageError::from)?);
        let src = RedbStore::from_database_arc(db.clone(), "src")?;
        let mut engine = FixtureEngine::default();
        src.store(b"k1", &mut engine, &1u32)?;
        src.store(b"k2", &mut engine, &2u32)?;

        let dst = RedbStore::from_database_arc(db, "dst")?;
        let stats = copy(&src, &dst)?;
        assert_eq!(stats.migrated, 2);

        assert_eq!(dst.fetch::<u32, _>(b"k1", &mut engine)?, Some(1));
        assert_eq!(dst.fetch::<u32, _>(b"k2", &mut engine)?, Some(2));
        Ok(())
    }

    #[cfg(feature = "redb")]
    #[test]
    fn redb_iter_encoded_empty() -> Result<()> {
        use storage::{RedbStore, StorageError};

        let file = tempfile::NamedTempFile::new().unwrap();
        let db = redb::Database::create(file.path()).map_err(StorageError::from)?;
        let store = RedbStore::from_database(db, "empty")?;
        let mut count = 0usize;
        store.iter_encoded(|_, _| {
            count += 1;
            Ok(())
        })?;
        assert_eq!(count, 0);
        Ok(())
    }

    // -- IterableStore: fjall2 ----------------------------------------------

    #[cfg(feature = "fjall2")]
    #[test]
    fn fjall2_iter_encoded_roundtrips() -> Result<()> {
        use storage::{Fjall2Store, StorageError};

        let tmp = tempfile::tempdir().unwrap();
        let ks = fjall2::Config::new(&tmp)
            .open()
            .map_err(StorageError::from)?;
        let src = Fjall2Store::open(ks.clone(), "src", None)?;
        let mut engine = FixtureEngine::default();
        src.store(b"k1", &mut engine, &1u32)?;
        src.store(b"k2", &mut engine, &2u32)?;

        let dst = Fjall2Store::open(ks, "dst", None)?;
        let stats = copy(&src, &dst)?;
        assert_eq!(stats.migrated, 2);

        assert_eq!(dst.fetch::<u32, _>(b"k1", &mut engine)?, Some(1));
        assert_eq!(dst.fetch::<u32, _>(b"k2", &mut engine)?, Some(2));
        Ok(())
    }

    #[cfg(feature = "fjall2")]
    #[test]
    fn fjall2_iter_encoded_empty() -> Result<()> {
        use storage::{Fjall2Store, StorageError};

        let tmp = tempfile::tempdir().unwrap();
        let ks = fjall2::Config::new(&tmp)
            .open()
            .map_err(StorageError::from)?;
        let store = Fjall2Store::open(ks, "empty", None)?;
        let mut count = 0usize;
        store.iter_encoded(|_, _| {
            count += 1;
            Ok(())
        })?;
        assert_eq!(count, 0);
        Ok(())
    }

    // -- IterableStore: fjall3 ----------------------------------------------

    #[cfg(feature = "fjall3")]
    #[test]
    fn fjall3_iter_encoded_roundtrips() -> Result<()> {
        use storage::{Fjall3Store, StorageError};

        let tmp = tempfile::tempdir().unwrap();
        let db = fjall3::Database::builder(&tmp)
            .open()
            .map_err(StorageError::from)?;
        let src = Fjall3Store::open(db, "src", None)?;
        let mut engine = FixtureEngine::default();
        src.store(b"k1", &mut engine, &1u32)?;
        src.store(b"k2", &mut engine, &2u32)?;

        let tmp2 = tempfile::tempdir().unwrap();
        let db2 = fjall3::Database::builder(&tmp2)
            .open()
            .map_err(StorageError::from)?;
        let dst = Fjall3Store::open(db2, "dst", None)?;
        let stats = copy(&src, &dst)?;
        assert_eq!(stats.migrated, 2);

        assert_eq!(dst.fetch::<u32, _>(b"k1", &mut engine)?, Some(1));
        assert_eq!(dst.fetch::<u32, _>(b"k2", &mut engine)?, Some(2));
        Ok(())
    }

    #[cfg(feature = "fjall3")]
    #[test]
    fn fjall3_iter_encoded_empty() -> Result<()> {
        use storage::{Fjall3Store, StorageError};

        let tmp = tempfile::tempdir().unwrap();
        let db = fjall3::Database::builder(&tmp)
            .open()
            .map_err(StorageError::from)?;
        let store = Fjall3Store::open(db, "empty", None)?;
        let mut count = 0usize;
        store.iter_encoded(|_, _| {
            count += 1;
            Ok(())
        })?;
        assert_eq!(count, 0);
        Ok(())
    }
}
