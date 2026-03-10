use super::{CacheStore, StorageResult, WorkerForkStore, private};

/// Fjall v2-backed cache store bound to a single partition.
pub struct Fjall2Store {
    handle: fjall2::PartitionHandle,
}

impl Fjall2Store {
    pub fn open(
        keyspace: fjall2::Keyspace,
        partition_name: impl AsRef<str>,
        options: Option<fjall2::PartitionCreateOptions>,
    ) -> StorageResult<Self> {
        let handle =
            keyspace.open_partition(partition_name.as_ref(), options.unwrap_or_default())?;
        Ok(Self { handle })
    }
}

impl private::Sealed for Fjall2Store {}

impl WorkerForkStore for Fjall2Store {
    fn fork_store(&self) -> Self {
        Self {
            handle: self.handle.clone(),
        }
    }
}

impl CacheStore for Fjall2Store {
    type Encoded<'a>
        = fjall2::UserValue
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
