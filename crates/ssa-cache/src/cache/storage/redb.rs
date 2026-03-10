use std::sync::Arc;

use redb::{ReadableDatabase, TableDefinition};

use super::{CacheStore, StorageError, StorageResult, WorkerForkStore, private};

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
#[derive(Debug)]
pub struct RedbStore {
    db: Arc<redb::Database>,
    table_name: &'static str,
}

impl RedbStore {
    pub fn new(db: redb::Database, table_name: &'static str) -> StorageResult<Self> {
        Self::from_arc(Arc::new(db), table_name)
    }

    pub fn from_arc(db: Arc<redb::Database>, table_name: &'static str) -> StorageResult<Self> {
        let tx = db.begin_write()?;
        {
            let table: TableDefinition<'static, &[u8], &[u8]> = TableDefinition::new(table_name);
            let _ = tx.open_table(table)?;
        }
        tx.commit()?;

        Ok(Self { db, table_name })
    }
}

impl private::Sealed for RedbStore {}

impl WorkerForkStore for RedbStore {
    fn fork_store(&self) -> Self {
        Self {
            db: self.db.clone(),
            table_name: self.table_name,
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
        let definition: TableDefinition<'static, &[u8], &[u8]> =
            TableDefinition::new(self.table_name);
        let table = tx.open_table(definition)?;
        let value = table.get(key)?;
        Ok(value.map(RedbEncoded))
    }

    fn store_encoded(&self, key: &[u8], encoded: &[u8]) -> StorageResult<()> {
        let tx = self.db.begin_write()?;

        {
            let definition: TableDefinition<'static, &[u8], &[u8]> =
                TableDefinition::new(self.table_name);
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
        let store = RedbStore::new(db, "test")?;
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
        let store = RedbStore::from_arc(db, "test")?;
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
            RedbStore::new(db, "test"),
            Err(StorageError::Redb(_))
        ));
        Ok(())
    }
}
