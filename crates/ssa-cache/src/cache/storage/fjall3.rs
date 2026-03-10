use std::sync::Arc;

use super::{CacheStore, Result, WorkerForkStore, private};

/// Fjall v3-backed cache store bound to a single keyspace.
pub struct Fjall3Store {
    handle: fjall3::Keyspace,
    _database: Arc<fjall3::Database>,
}

impl Fjall3Store {
    pub fn open(
        database: fjall3::Database,
        keyspace_name: impl AsRef<str>,
        options: Option<fjall3::KeyspaceCreateOptions>,
    ) -> Result<Self> {
        Self::open_arc(Arc::new(database), keyspace_name, options)
    }

    pub fn open_arc(
        database: Arc<fjall3::Database>,
        keyspace_name: impl AsRef<str>,
        options: Option<fjall3::KeyspaceCreateOptions>,
    ) -> Result<Self> {
        let create_options = options.unwrap_or_default();
        let handle = database.keyspace(keyspace_name.as_ref(), || create_options)?;
        Ok(Self {
            handle,
            _database: database,
        })
    }
}

impl private::Sealed for Fjall3Store {}

impl WorkerForkStore for Fjall3Store {
    fn fork_store(&self) -> Self {
        Self {
            handle: self.handle.clone(),
            _database: self._database.clone(),
        }
    }
}

impl CacheStore for Fjall3Store {
    fn fetch_encoded_with<T, E, F>(&self, key: &[u8], f: F) -> std::result::Result<T, E>
    where
        F: FnOnce(Option<&[u8]>) -> std::result::Result<T, E>,
        E: From<super::Error>,
    {
        let value = self
            .handle
            .get(key)
            .map_err(super::Error::from)
            .map_err(E::from)?;
        f(value.as_ref().map(|bytes| bytes.as_ref()))
    }

    fn store_encoded(&self, key: &[u8], encoded: &[u8]) -> Result<()> {
        self.handle.insert(key, encoded)?;
        Ok(())
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::{
        cache::{codec::fixtures::FixtureEngine, storage::Error as StorageError},
        error::Result,
    };

    #[test]
    fn test_fjall3_store() -> Result<()> {
        let tmp = tempfile::tempdir().unwrap();
        let db = ::fjall3::Database::builder(&tmp)
            .open()
            .map_err(StorageError::from)?;
        let store = Fjall3Store::open(db, "test", None)?;
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
