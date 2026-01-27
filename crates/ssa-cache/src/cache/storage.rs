use std::sync::{Arc, RwLock};

use crate::{
    Result,
    cache::codec::{Decode, Encode},
};

/// A trait defining a cache storage interface for storing and retrieving cacheable values.
pub trait CacheStore<E>: Sync {
    /// Attempts to fetch a value from the cache.
    ///
    /// # Arguments
    ///
    /// - `sig` - The signature (key) used to identify the cached value,
    /// - `engine` - A codec engine for deserialization.
    ///
    /// # Returns
    ///
    /// If the value was found and successfully decoded, return `Some(T)`.
    /// If no value was found for the given signature, return `None`.
    ///
    /// # Errors
    ///
    /// If there are some errors during deserialization or storage access.
    fn fetch<T>(&self, sig: &[u8], engine: &mut E) -> Result<Option<T>>
    where
        E: Decode<T>;

    /// Stores a value in the cache.
    ///
    /// # Arguments
    ///
    /// - `sig` - The signature (key) to associate with the value,
    /// - `engine` - A codec engine for serialization,
    /// - `value` - The value to be cached.
    ///
    /// # Errors
    ///
    /// If there are some errors during serialization or storage access.
    fn store<T>(&self, sig: &[u8], engine: &mut E, value: &T) -> Result<()>
    where
        E: Encode<T>;

    fn fetch_or_execute<T, F>(&self, sig: &[u8], engine: &mut E, execute: F) -> Result<T>
    where
        E: Encode<T> + Decode<T>,
        F: FnOnce() -> Result<T>,
    {
        if let Some(cached) = self.fetch(sig, engine)? {
            Ok(cached)
        } else {
            let output = execute()?;
            self.store(sig, engine, &output)?;
            Ok(output)
        }
    }
}

impl<E> CacheStore<E> for () {
    fn fetch<T>(&self, _: &[u8], _: &mut E) -> Result<Option<T>>
    where
        E: Decode<T>,
    {
        Ok(None)
    }

    fn store<T>(&self, _: &[u8], _: &mut E, _: &T) -> Result<()>
    where
        E: Encode<T>,
    {
        Ok(())
    }
}

type HashMap<H> = std::collections::HashMap<Vec<u8>, Vec<u8>, H>;

#[derive(Debug, Default)]
pub struct HashMapStore<H>(Arc<RwLock<HashMap<H>>>);

impl<E, H> CacheStore<E> for HashMapStore<H>
where
    H: std::hash::BuildHasher + Send + Sync,
{
    fn fetch<T>(&self, sig: &[u8], engine: &mut E) -> Result<Option<T>>
    where
        E: Decode<T>,
    {
        let map = self.0.read().unwrap();
        map.get(sig).map(|value| engine.decode(value)).transpose()
    }

    fn store<T>(&self, sig: &[u8], engine: &mut E, value: &T) -> Result<()>
    where
        E: Encode<T>,
    {
        let value = engine.encode(value);
        let mut map = self.0.write().unwrap();
        map.insert(sig.to_owned(), value.to_owned());
        Ok(())
    }
}

#[cfg(feature = "fjall")]
mod fjall_store {
    use super::*;

    impl<E> CacheStore<E> for fjall::Keyspace {
        fn fetch<T>(&self, sig: &[u8], engine: &mut E) -> Result<Option<T>>
        where
            E: Decode<T>,
        {
            let value = self.get(sig)?;
            value
                .as_ref()
                .map(|value| engine.decode(value.as_ref()))
                .transpose()
        }

        fn store<T>(&self, sig: &[u8], engine: &mut E, value: &T) -> Result<()>
        where
            E: Encode<T>,
        {
            let value = engine.encode(value);
            self.insert(sig, value)?;
            Ok(())
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use std::collections::hash_map::RandomState;

    use super::*;

    type Engine = crate::DefaultCodec;

    #[test]
    fn test_hashmap_store() -> Result<()> {
        let store = HashMapStore::<RandomState>::default();
        let mut engine = Engine::default();

        // Test storing and fetching a value
        let value = 42u32;
        let sig = b"test_sig";
        store.store(sig, &mut engine, &value)?;
        assert_eq!(store.fetch(sig, &mut engine)?, Some(value));

        // Test fetching non-existent key
        assert_eq!(store.fetch::<u32>(b"non_existent", &mut engine)?, None);

        // Test fetching key with wrong type
        assert!(matches!(
            store.fetch::<u64>(sig, &mut engine),
            Err(crate::Error::BitCode(_))
        ));

        Ok(())
    }

    #[cfg(feature = "fjall")]
    #[test]
    fn test_fjall_store() -> Result<()> {
        let tmp = tempfile::tempdir().unwrap();
        let db = fjall::Database::builder(&tmp).open()?;
        let partition = db.keyspace("test", Default::default)?;
        let mut engine = Engine::default();

        // Test storing and fetching a value
        let value = 42u32;
        let sig = b"test_sig";
        partition.store(sig, &mut engine, &value)?;
        assert_eq!(partition.fetch(sig, &mut engine)?, Some(value));

        // Test fetching non-existent key
        assert_eq!(partition.fetch::<u32>(b"non_existent", &mut engine)?, None);

        // Test fetching key with wrong type
        assert!(matches!(
            partition.fetch::<u64>(sig, &mut engine),
            Err(crate::Error::BitCode(_))
        ));

        Ok(())
    }
}
