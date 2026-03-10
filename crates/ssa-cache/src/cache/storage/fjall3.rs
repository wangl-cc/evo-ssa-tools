use super::{CacheStore, StorageResult, WorkerForkStore, private};

/// Fjall v3-backed cache store bound to a single keyspace.
pub struct Fjall3Store {
    handle: fjall3::Keyspace,
}

impl Fjall3Store {
    pub fn open(
        database: fjall3::Database,
        keyspace_name: impl AsRef<str>,
        options: Option<fjall3::KeyspaceCreateOptions>,
    ) -> StorageResult<Self> {
        let create_options = options.unwrap_or_default();
        let handle = database.keyspace(keyspace_name.as_ref(), || create_options)?;
        Ok(Self { handle })
    }
}

impl private::Sealed for Fjall3Store {}

impl WorkerForkStore for Fjall3Store {
    fn fork_store(&self) -> Self {
        Self {
            handle: self.handle.clone(),
        }
    }
}

impl CacheStore for Fjall3Store {
    type Encoded<'a>
        = fjall3::UserValue
    where
        Self: 'a;

    fn fetch_encoded(&self, key: &[u8]) -> StorageResult<Option<Self::Encoded<'_>>> {
        Ok(self.handle.get(key)?)
    }

    fn store_encoded(&self, key: &[u8], encoded: &[u8]) -> StorageResult<()> {
        self.handle.insert(key, encoded)?;
        Ok(())
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::{
        cache::{codec::fixtures::FixtureEngine, storage::StorageError},
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

    #[test]
    fn test_fjall3_store_fetch_encoded_and_fork() -> Result<()> {
        let tmp = tempfile::tempdir().unwrap();
        let db = ::fjall3::Database::builder(&tmp)
            .open()
            .map_err(StorageError::from)?;
        let store = Fjall3Store::open(db, "raw", None)?;
        let forked = store.fork_store();

        store.store_encoded(b"k", b"payload")?;

        let encoded = forked.fetch_encoded(b"k")?.expect("value should exist");
        assert_eq!(encoded.as_ref(), b"payload");
        assert!(forked.fetch_encoded(b"missing")?.is_none());
        Ok(())
    }
}
