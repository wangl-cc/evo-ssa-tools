use std::sync::{Arc, RwLock};

use crate::Result;

pub trait CodecBuffer {
    fn init() -> Self;
}

pub trait Encodeable {
    type Buffer: CodecBuffer;

    fn encode<'b>(&self, buffer: &'b mut Self::Buffer) -> &'b [u8];
}

pub trait Cacheable: Encodeable + Sized {
    fn decode(bytes: &[u8], buffer: &mut Self::Buffer) -> Result<Self>;
}

#[cfg(feature = "bitcode")]
impl CodecBuffer for bitcode::Buffer {
    fn init() -> Self {
        Self::new()
    }
}

#[cfg(feature = "bitcode")]
mod bitcode_codec {
    use super::*;

    impl<T> Encodeable for T
    where
        T: bitcode::Encode,
    {
        type Buffer = bitcode::Buffer;

        fn encode<'b>(&self, buffer: &'b mut Self::Buffer) -> &'b [u8] {
            buffer.encode(self)
        }
    }

    impl<T> Cacheable for T
    where
        T: bitcode::Encode + for<'b> bitcode::Decode<'b>,
    {
        fn decode(bytes: &[u8], buffer: &mut Self::Buffer) -> Result<Self> {
            Ok(buffer.decode(bytes)?)
        }
    }
}

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
    fn fetch<T: Cacheable>(&self, sig: &[u8], buffer: &mut T::Buffer) -> Result<Option<T>>;

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
    fn store<T: Cacheable>(&self, sig: &[u8], buffer: &mut T::Buffer, value: &T) -> Result<()>;
}

impl CacheStore for () {
    fn fetch<T: Cacheable>(&self, _sig: &[u8], _buffer: &mut T::Buffer) -> Result<Option<T>> {
        Ok(None)
    }

    fn store<T: Cacheable>(&self, _sig: &[u8], _buffer: &mut T::Buffer, _value: &T) -> Result<()> {
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
    fn fetch<T: Cacheable>(&self, sig: &[u8], buffer: &mut T::Buffer) -> Result<Option<T>> {
        let map = self.0.read().unwrap();
        map.get(sig)
            .map(|value| T::decode(value.as_slice(), buffer))
            .transpose()
    }

    fn store<T: Cacheable>(&self, sig: &[u8], buffer: &mut T::Buffer, value: &T) -> Result<()> {
        let value = value.encode(buffer).to_vec();
        let mut map = self.0.write().unwrap();
        map.insert(sig.to_owned(), value);
        Ok(())
    }
}

#[cfg(feature = "fjall")]
mod fjall_store {
    use super::*;

    impl CacheStore for fjall::Partition {
        fn fetch<T: Cacheable>(&self, sig: &[u8], buffer: &mut T::Buffer) -> Result<Option<T>> {
            let value = self.get(sig)?;
            value
                .map(|value| T::decode(value.as_ref(), buffer))
                .transpose()
        }

        fn store<T: Cacheable>(&self, sig: &[u8], buffer: &mut T::Buffer, value: &T) -> Result<()> {
            let value = value.encode(buffer);
            self.insert(sig, value)?;
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::hash_map::RandomState;

    use super::*;

    #[test]
    fn test_no_store() -> Result<()> {
        let store = ();
        let mut buffer = bitcode::Buffer::new();

        // Test storing and fetching a value
        let value = 42u32;
        let sig = b"test_sig";
        store.store(sig, &mut buffer, &value)?;
        assert_eq!(store.fetch::<u32>(sig, &mut buffer)?, None);

        Ok(())
    }

    #[test]
    fn test_hashmap_store() -> Result<()> {
        let store = HashMapStore::<RandomState>::default();
        let mut buffer = bitcode::Buffer::new();

        // Test storing and fetching a value
        let value = 42u32;
        let sig = b"test_sig";
        store.store(sig, &mut buffer, &value)?;
        assert_eq!(store.fetch(sig, &mut buffer)?, Some(value));

        // Test fetching non-existent key
        assert_eq!(store.fetch::<u32>(b"non_existent", &mut buffer)?, None);

        Ok(())
    }

    #[cfg(feature = "fjall")]
    #[test]
    fn test_fjall_store() -> Result<()> {
        let db = fjall::Config::new(tempfile::tempdir().unwrap()).open()?;
        let partition = db.open_partition("test", Default::default())?;
        let mut buffer = bitcode::Buffer::new();

        // Test storing and fetching a value
        let value = 42u32;
        let sig = b"test_sig";
        partition.store(sig, &mut buffer, &value)?;
        assert_eq!(partition.fetch(sig, &mut buffer)?, Some(value));

        // Test fetching non-existent key
        assert_eq!(partition.fetch::<u32>(b"non_existent", &mut buffer)?, None);

        Ok(())
    }
}
