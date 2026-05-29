//! Fjall v3 storage provider and raw store helpers.

use super::{EncodedStorage, StorageNamespace, StorageResult};
use crate::{Result, cache::StorageProvider};

/// Fjall v3-backed raw store bound to a single keyspace.
///
/// The caller owns the surrounding [`fjall3::Database`] handle and decides how different cache
/// keyspaces map onto Fjall keyspace names.
///
/// All workers share the same keyspace handle: a value written by one worker is immediately
/// visible to the others.
pub struct Fjall3Store {
    pub(crate) handle: fjall3::Keyspace,
}

impl Fjall3Store {
    /// Open or create a keyspace-backed store inside an existing Fjall v3 database.
    pub fn open(database: fjall3::Database, keyspace_name: impl AsRef<str>) -> StorageResult<Self> {
        Self::open_with_options(
            database,
            keyspace_name,
            fjall3::KeyspaceCreateOptions::default(),
        )
    }

    /// Open or create a keyspace-backed store with custom keyspace creation options.
    pub fn open_with_options(
        database: fjall3::Database,
        keyspace_name: impl AsRef<str>,
        create_options: fjall3::KeyspaceCreateOptions,
    ) -> StorageResult<Self> {
        let handle = database.keyspace(keyspace_name.as_ref(), || create_options)?;
        Ok(Self { handle })
    }
}

/// Fjall v3 storage provider for managed cache spaces.
///
/// The provider owns a database handle and the keyspace options used when opening namespaces.
/// Multiple providers can share the same database while using different keyspace options.
#[derive(Clone)]
pub struct Fjall3StorageProvider {
    database: fjall3::Database,
    keyspace_options: fjall3::KeyspaceCreateOptions,
}

impl Fjall3StorageProvider {
    /// Create a storage provider from an existing Fjall v3 database.
    pub fn new(database: fjall3::Database) -> Self {
        Self {
            database,
            keyspace_options: fjall3::KeyspaceCreateOptions::default(),
        }
    }

    /// Set keyspace creation options used for newly opened namespaces.
    pub fn with_keyspace_options(mut self, options: fjall3::KeyspaceCreateOptions) -> Self {
        self.keyspace_options = options;
        self
    }
}

impl StorageProvider for Fjall3StorageProvider {
    type Storage = Fjall3Store;

    fn open_storage(&self, namespace: &StorageNamespace) -> Result<Self::Storage> {
        Ok(Fjall3Store::open_with_options(
            self.database.clone(),
            namespace.as_str(),
            self.keyspace_options.clone(),
        )?)
    }
}

#[cfg(not(test))]
impl super::sealed::Sealed for Fjall3Store {}

impl crate::cache::CloneShared for Fjall3Store {
    fn clone_shared(&self) -> Self {
        Self {
            handle: self.handle.clone(),
        }
    }
}

impl EncodedStorage for Fjall3Store {
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
        cache::{
            Cache, CloneShared, EncodedCache, StorageProviderExt, codec::fixtures::FixtureEngine,
            provider::CacheProvider, storage::StorageError,
        },
        error::Result,
        identity::ComputationPath,
    };

    #[test]
    fn test_fjall3_store() -> Result<()> {
        let tmp = tempfile::tempdir().unwrap();
        let db = ::fjall3::Database::builder(&tmp)
            .open()
            .map_err(StorageError::from)?;
        let store = Fjall3Store::open(db, "test")?;
        let mut cache = EncodedCache::new(store, FixtureEngine::default());

        assert_eq!(cache.fetch(b"non_existent")?, None::<u32>);

        cache.store(b"k", &42u32)?;
        assert_eq!(cache.fetch(b"k")?, Some(42u32));

        let mut mismatched_cache =
            EncodedCache::new(cache.storage.clone_shared(), FixtureEngine::default());
        assert!(crate::cache::Cache::<u64>::fetch(&mut mismatched_cache, b"k").is_err());
        Ok(())
    }

    #[test]
    fn test_fjall3_store_fetch_encoded_and_clone_shared() -> Result<()> {
        let tmp = tempfile::tempdir().unwrap();
        let db = ::fjall3::Database::builder(&tmp)
            .open()
            .map_err(StorageError::from)?;
        let store = Fjall3Store::open(db, "raw")?;
        let forked = store.clone_shared();

        store.store_encoded(b"k", b"payload")?;

        let encoded = forked.fetch_encoded(b"k")?.expect("value should exist");
        assert_eq!(encoded.as_ref(), b"payload");
        assert!(forked.fetch_encoded(b"missing")?.is_none());
        Ok(())
    }

    #[test]
    fn test_fjall3_storage_provider_opens_distinct_namespaces() -> Result<()> {
        let tmp = tempfile::tempdir().unwrap();
        let db = ::fjall3::Database::builder(&tmp)
            .open()
            .map_err(StorageError::from)?;
        let provider = Fjall3StorageProvider::new(db)
            .with_keyspace_options(fjall3::KeyspaceCreateOptions::default())
            .with_codec(FixtureEngine::default());
        let path_a = ComputationPath::root_from_str("a-v1");
        let path_b = ComputationPath::root_from_str("b-v1");
        let mut cache_a = <_ as CacheProvider<u32>>::bind(provider.clone(), &path_a)?;
        let mut cache_b = <_ as CacheProvider<u32>>::bind(provider, &path_b)?;

        assert_eq!(cache_a.fetch_or_execute(b"k", || Ok(1))?, 1);
        assert_eq!(cache_b.fetch_or_execute(b"k", || Ok(2))?, 2);
        assert_eq!(cache_a.fetch_or_execute(b"k", || Ok(3))?, 1);
        assert_eq!(cache_b.fetch_or_execute(b"k", || Ok(4))?, 2);
        Ok(())
    }
}
