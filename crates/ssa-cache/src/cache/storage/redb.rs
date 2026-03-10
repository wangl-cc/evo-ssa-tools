use std::sync::Arc;

use redb::{ReadableDatabase, TableDefinition};

use super::{CacheStore, Error, Result, WorkerForkStore, private};

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

impl From<RedbError> for Error {
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

impl From<redb::CommitError> for Error {
    fn from(error: redb::CommitError) -> Self {
        RedbError::from(error).into()
    }
}

impl From<redb::DatabaseError> for Error {
    fn from(error: redb::DatabaseError) -> Self {
        RedbError::from(error).into()
    }
}

impl From<redb::StorageError> for Error {
    fn from(error: redb::StorageError) -> Self {
        RedbError::from(error).into()
    }
}

impl From<redb::TableError> for Error {
    fn from(error: redb::TableError) -> Self {
        RedbError::from(error).into()
    }
}

impl From<redb::TransactionError> for Error {
    fn from(error: redb::TransactionError) -> Self {
        RedbError::from(error).into()
    }
}

/// `redb`-backed cache store bound to a single table.
#[derive(Debug)]
pub struct RedbStore {
    db: Arc<redb::Database>,
    table_name: &'static str,
}

impl RedbStore {
    pub fn new(db: redb::Database, table_name: &'static str) -> Result<Self> {
        Self::from_arc(Arc::new(db), table_name)
    }

    pub fn from_arc(db: Arc<redb::Database>, table_name: &'static str) -> Result<Self> {
        let tx = db.begin_write().map_err(Error::from)?;
        {
            let table: TableDefinition<'static, &[u8], &[u8]> = TableDefinition::new(table_name);
            let _ = tx.open_table(table).map_err(Error::from)?;
        }
        tx.commit().map_err(Error::from)?;

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
    fn fetch_encoded_with<T, E, F>(&self, key: &[u8], f: F) -> std::result::Result<T, E>
    where
        F: FnOnce(Option<&[u8]>) -> std::result::Result<T, E>,
        E: From<super::Error>,
    {
        let tx = self.db.begin_read().map_err(Error::from).map_err(E::from)?;
        let table: TableDefinition<'static, &[u8], &[u8]> = TableDefinition::new(self.table_name);

        match tx.open_table(table) {
            Ok(table) => match table.get(key).map_err(Error::from).map_err(E::from)? {
                Some(value) => f(Some(value.value())),
                None => f(None),
            },
            Err(redb::TableError::TableDoesNotExist(_)) => f(None),
            Err(error) => Err(E::from(Error::from(error))),
        }
    }

    fn store_encoded(&self, key: &[u8], encoded: &[u8]) -> Result<()> {
        let tx = self.db.begin_write().map_err(Error::from)?;

        {
            let table: TableDefinition<'static, &[u8], &[u8]> =
                TableDefinition::new(self.table_name);
            let mut table = tx.open_table(table).map_err(Error::from)?;
            table.insert(key, encoded).map_err(Error::from)?;
        }

        tx.commit().map_err(Error::from)?;
        Ok(())
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::{cache::codec::fixtures::FixtureEngine, error::Result};

    #[test]
    fn test_redb_store() -> Result<()> {
        let file = tempfile::NamedTempFile::new().unwrap();
        let db = ::redb::Database::create(file.path()).map_err(Error::from)?;
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
