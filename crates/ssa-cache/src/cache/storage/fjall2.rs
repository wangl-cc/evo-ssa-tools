use std::sync::Arc;

use super::{CacheStore, Result, WorkerForkStore, private};

/// Fjall v2-backed cache store bound to a single partition.
pub struct Fjall2Store {
    handle: fjall2::PartitionHandle,
    _keyspace: Arc<fjall2::Keyspace>,
}

impl Fjall2Store {
    pub fn open(
        keyspace: fjall2::Keyspace,
        partition_name: impl AsRef<str>,
        options: Option<fjall2::PartitionCreateOptions>,
    ) -> Result<Self> {
        Self::open_arc(Arc::new(keyspace), partition_name, options)
    }

    pub fn open_arc(
        keyspace: Arc<fjall2::Keyspace>,
        partition_name: impl AsRef<str>,
        options: Option<fjall2::PartitionCreateOptions>,
    ) -> Result<Self> {
        let handle =
            keyspace.open_partition(partition_name.as_ref(), options.unwrap_or_default())?;
        Ok(Self {
            handle,
            _keyspace: keyspace,
        })
    }
}

impl private::Sealed for Fjall2Store {}

impl WorkerForkStore for Fjall2Store {
    fn fork_store(&self) -> Self {
        Self {
            handle: self.handle.clone(),
            _keyspace: self._keyspace.clone(),
        }
    }
}

impl CacheStore for Fjall2Store {
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
    fn test_fjall2_store() -> Result<()> {
        let tmp = tempfile::tempdir().unwrap();
        let db = ::fjall2::Config::new(&tmp)
            .open()
            .map_err(StorageError::from)?;
        let store = Fjall2Store::open(db, "test", None)?;
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
