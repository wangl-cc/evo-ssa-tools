//! Persistent storage backend built on [`redb`].

use std::{path::Path, sync::Arc};

use redb::{ReadableDatabase, TableDefinition};

use super::{CacheStore, StorageError, StorageResult, WorkerForkStore, private};

type BytesTable<'name> = TableDefinition<'name, &'static [u8], &'static [u8]>;

macro_rules! impl_storage_error_from_redb {
    ($($ty:path),* $(,)?) => {
        $(
            impl From<$ty> for StorageError {
                fn from(error: $ty) -> Self {
                    Self::Redb(error.into())
                }
            }
        )*
    };
}

impl_storage_error_from_redb!(
    redb::CommitError,
    redb::DatabaseError,
    redb::StorageError,
    redb::TableError,
    redb::TransactionError,
);

#[doc(hidden)]
pub struct RedbEncoded<'a>(redb::AccessGuard<'a, &'static [u8]>);

impl AsRef<[u8]> for RedbEncoded<'_> {
    fn as_ref(&self) -> &[u8] {
        self.0.value()
    }
}

/// `redb`-backed cache store bound to a single table.
///
/// Each cache operation opens a read or write transaction against `table_name`. Use distinct table
/// names for distinct cache semantics, even if the key and value byte formats happen to match.
#[derive(Debug)]
pub struct RedbStore {
    pub(crate) db: Arc<redb::Database>,
    pub(crate) table_name: String,
}

impl RedbStore {
    /// Open a database file and bind this store to `table_name`.
    ///
    /// This uses [`redb::Database::create`], so an empty or missing file is initialized on first
    /// use and an existing valid redb file is opened.
    pub fn open(path: impl AsRef<Path>, table_name: impl Into<String>) -> StorageResult<Self> {
        let db = redb::Database::create(path)?;
        Self::from_database(db, table_name)
    }

    /// Create a store from an owned [`redb::Database`] handle.
    ///
    /// The target table is opened during construction so configuration and type mismatches fail
    /// early.
    pub fn from_database(db: redb::Database, table_name: impl Into<String>) -> StorageResult<Self> {
        Self::from_database_arc(Arc::new(db), table_name)
    }

    /// Create a store from a shared [`Arc<redb::Database>`] handle.
    ///
    /// This is useful when multiple caches or subsystems need to share the same database file
    /// while using different table names.
    pub fn from_database_arc(
        db: Arc<redb::Database>,
        table_name: impl Into<String>,
    ) -> StorageResult<Self> {
        let table_name = table_name.into();
        Self::init_table(&db, &table_name)?;
        Ok(Self { db, table_name })
    }

    fn init_table(db: &redb::Database, table_name: &str) -> StorageResult<()> {
        let tx = db.begin_write()?;
        {
            let table = BytesTable::new(table_name);
            // Try to open the table, creating it if it doesn't exist.
            tx.open_table(table)?;
        }
        tx.commit()?;
        Ok(())
    }
}

impl private::Sealed for RedbStore {}

impl WorkerForkStore for RedbStore {
    fn fork_store(&self) -> Self {
        Self {
            db: self.db.clone(),
            table_name: self.table_name.clone(),
        }
    }
}

impl CacheStore for RedbStore {
    type Encoded<'a>
        = RedbEncoded<'a>
    where
        Self: 'a;

    fn fetch_encoded(&self, key: &[u8]) -> StorageResult<Option<Self::Encoded<'_>>> {
        let tx = self.db.begin_read()?;
        let definition = BytesTable::new(self.table_name.as_ref());
        let table = tx.open_table(definition)?;
        let value = table.get(key)?;
        Ok(value.map(RedbEncoded))
    }

    fn store_encoded(&self, key: &[u8], encoded: &[u8]) -> StorageResult<()> {
        let tx = self.db.begin_write()?;

        {
            let definition = BytesTable::new(self.table_name.as_ref());
            let mut table = tx.open_table(definition)?;
            table.insert(key, encoded)?;
        }

        tx.commit()?;
        Ok(())
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::{cache::codec::fixtures::FixtureEngine, error::Result as CrateResult};

    const REDB_BYTES_TABLE: redb::TableDefinition<&[u8], u64> = redb::TableDefinition::new("test");

    #[test]
    fn test_redb_store() -> CrateResult<()> {
        let file = tempfile::NamedTempFile::new().unwrap();
        let db = ::redb::Database::create(file.path()).map_err(StorageError::from)?;
        let store = RedbStore::from_database(db, "test")?;
        let mut engine = FixtureEngine::default();

        assert_eq!(
            store.fetch::<u32, FixtureEngine>(b"non_existent", &mut engine)?,
            None
        );

        store.store::<u32, FixtureEngine>(b"k", &mut engine, &42u32)?;
        assert_eq!(
            store.fetch::<u32, FixtureEngine>(b"k", &mut engine)?,
            Some(42)
        );
        assert!(
            store
                .fetch::<u64, FixtureEngine>(b"k", &mut engine)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn test_redb_store_fetch_encoded_and_fork() -> CrateResult<()> {
        let file = tempfile::NamedTempFile::new().unwrap();
        let db = Arc::new(::redb::Database::create(file.path()).map_err(StorageError::from)?);
        let store = RedbStore::from_database_arc(db, "test")?;
        let forked = store.fork_store();

        store.store_encoded(b"raw", b"payload")?;

        let encoded = forked.fetch_encoded(b"raw")?.expect("value should exist");
        assert_eq!(encoded.as_ref(), b"payload");
        assert!(forked.fetch_encoded(b"missing")?.is_none());

        Ok(())
    }

    #[test]
    fn test_redb_store_open_type_mismatch_returns_storage_error() -> CrateResult<()> {
        let file = tempfile::NamedTempFile::new().unwrap();
        let db = ::redb::Database::create(file.path()).map_err(StorageError::from)?;

        let tx = db.begin_write().map_err(StorageError::from)?;
        {
            let mut table = tx
                .open_table(REDB_BYTES_TABLE)
                .map_err(StorageError::from)?;
            table
                .insert(b"k".as_slice(), &1u64)
                .map_err(StorageError::from)?;
        }
        tx.commit().map_err(StorageError::from)?;

        assert!(matches!(
            RedbStore::from_database(db, "test"),
            Err(StorageError::Redb(_))
        ));
        Ok(())
    }

    #[test]
    fn test_redb_store_open_creates_and_reopens_database_file() -> CrateResult<()> {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cache.redb");

        let mut engine = FixtureEngine::default();
        {
            let store = RedbStore::open(&path, "test")?;
            store.store::<u32, FixtureEngine>(b"k", &mut engine, &7u32)?;
        }

        let reopened = RedbStore::open(&path, "test")?;
        assert_eq!(
            reopened.fetch::<u32, FixtureEngine>(b"k", &mut engine)?,
            Some(7)
        );
        Ok(())
    }
}
