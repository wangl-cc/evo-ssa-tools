use std::sync::{Arc, RwLock};

use crate::{Result, cache::codec::Codec};

/// A trait defining a cache storage interface for storing and retrieving cacheable values.
pub trait CacheStore: Sync {
    /// Attempts to fetch a value from the cache.
    ///
    /// # Arguments
    ///
    /// - `sig` - The signature (key) used to identify the cached value,
    /// - `buffer` - A reusable buffer for deserialization.
    ///
    /// # Returns
    ///
    /// If the value was found and successfully decoded, return `Some(T)`.
    /// If no value was found for the given signature, return `None`.
    ///
    /// # Errors
    ///
    /// If there are some errors during deserialization or storage access.
    fn fetch<T: Codec>(&self, sig: &[u8], buffer: &mut T::Buffer) -> Result<Option<T>>;

    /// Stores a value in the cache.
    ///
    /// # Arguments
    ///
    /// - `sig` - The signature (key) to associate with the value,
    /// - `buffer` - A reusable buffer for serialization,
    /// - `value` - The value to be cached.
    ///
    /// # Errors
    ///
    /// If there are some errors during serialization or storage access.
    fn store<T: Codec>(&self, sig: &[u8], buffer: &mut T::Buffer, value: &T) -> Result<()>;

    fn fetch_or_execute<T, F>(&self, sig: &[u8], buffer: &mut T::Buffer, execute: F) -> Result<T>
    where
        T: Codec,
        F: FnOnce() -> Result<T>,
    {
        if let Some(cached) = self.fetch(sig, buffer)? {
            Ok(cached)
        } else {
            let output = execute()?;
            self.store(sig, buffer, &output)?;
            Ok(output)
        }
    }
}

impl CacheStore for () {
    fn fetch<T: Codec>(&self, _: &[u8], _: &mut T::Buffer) -> Result<Option<T>> {
        Ok(None)
    }

    fn store<T: Codec>(&self, _: &[u8], _: &mut T::Buffer, _: &T) -> Result<()> {
        Ok(())
    }
}

type HashMap<H> = std::collections::HashMap<Vec<u8>, Vec<u8>, H>;

#[derive(Debug, Default)]
pub struct HashMapStore<H>(Arc<RwLock<HashMap<H>>>);

impl<H> CacheStore for HashMapStore<H>
where
    H: std::hash::BuildHasher + Send + Sync,
{
    fn fetch<T: Codec>(&self, sig: &[u8], buffer: &mut T::Buffer) -> Result<Option<T>> {
        let map = self.0.read().unwrap();
        map.get(sig)
            .map(|value| T::decode(value.as_slice(), buffer))
            .transpose()
    }

    fn store<T: Codec>(&self, sig: &[u8], buffer: &mut T::Buffer, value: &T) -> Result<()> {
        let value = value.encode(buffer);
        let mut map = self.0.write().unwrap();
        map.insert(sig.to_owned(), value.to_vec());
        Ok(())
    }
}

#[cfg(feature = "fjall")]
mod fjall_store {
    use super::*;

    impl CacheStore for fjall::Keyspace {
        fn fetch<T: Codec>(&self, sig: &[u8], buffer: &mut T::Buffer) -> Result<Option<T>> {
            let value = self.get(sig)?;
            value
                .map(|value| T::decode(value.as_ref(), buffer))
                .transpose()
        }

        fn store<T: Codec>(&self, sig: &[u8], buffer: &mut T::Buffer, value: &T) -> Result<()> {
            let value = value.encode(buffer);
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
    use crate::cache::codec::CodecBuffer;

    #[test]
    fn test_hashmap_store() -> Result<()> {
        let store = HashMapStore::<RandomState>::default();
        let mut buffer = <u32 as Codec>::Buffer::init();

        // Test storing and fetching a value
        let value = 42u32;
        let sig = b"test_sig";
        store.store(sig, &mut buffer, &value)?;
        assert_eq!(store.fetch(sig, &mut buffer)?, Some(value));

        // Test fetching non-existent key
        assert_eq!(store.fetch::<u32>(b"non_existent", &mut buffer)?, None);

        // Test fetching key with wrong type
        assert!(matches!(
            store.fetch::<u64>(sig, &mut buffer),
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
        let mut buffer = <u32 as Codec>::Buffer::init();

        // Test storing and fetching a value
        let value = 42u32;
        let sig = b"test_sig";
        partition.store(sig, &mut buffer, &value)?;
        assert_eq!(partition.fetch(sig, &mut buffer)?, Some(value));

        // Test fetching non-existent key
        assert_eq!(partition.fetch::<u32>(b"non_existent", &mut buffer)?, None);

        // Test fetching key with wrong type
        assert!(matches!(
            partition.fetch::<u64>(sig, &mut buffer),
            Err(crate::Error::BitCode(_))
        ));

        Ok(())
    }
}
