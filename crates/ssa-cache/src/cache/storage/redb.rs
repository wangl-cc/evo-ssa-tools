use std::sync::Arc;

use redb::{ReadableDatabase, TableDefinition};

use super::{CacheStore, StorageError, StorageResult, WorkerForkStore, private};

#[derive(thiserror::Error, Debug)]
enum RedbError {
    #[error(transparent)]
    Commit(#[from] redb::CommitError),

    #[error(transparent)]
    Database(#[from] redb::DatabaseError),

    #[error(transparent)]
    Storage(#[from] redb::StorageError),

    #[error(transparent)]
    Table(#[from] redb::TableError),

    #[error(transparent)]
    Transaction(#[from] redb::TransactionError),
}

impl From<RedbError> for StorageError {
    fn from(error: RedbError) -> Self {
        match error {
            RedbError::Commit(error) => Self::Redb(error.into()),
            RedbError::Database(error) => Self::Redb(error.into()),
            RedbError::Storage(error) => Self::Redb(error.into()),
            RedbError::Table(error) => Self::Redb(error.into()),
            RedbError::Transaction(error) => Self::Redb(error.into()),
        }
    }
}

impl From<redb::CommitError> for StorageError {
    fn from(error: redb::CommitError) -> Self {
        RedbError::from(error).into()
    }
}

impl From<redb::DatabaseError> for StorageError {
    fn from(error: redb::DatabaseError) -> Self {
        RedbError::from(error).into()
    }
}

impl From<redb::StorageError> for StorageError {
    fn from(error: redb::StorageError) -> Self {
        RedbError::from(error).into()
    }
}

impl From<redb::TableError> for StorageError {
    fn from(error: redb::TableError) -> Self {
        RedbError::from(error).into()
    }
}

impl From<redb::TransactionError> for StorageError {
    fn from(error: redb::TransactionError) -> Self {
        RedbError::from(error).into()
    }
}

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
}
