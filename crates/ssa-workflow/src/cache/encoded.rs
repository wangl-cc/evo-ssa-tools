//! Typed cache adapter backed by raw [`EncodedStorage`](crate::cache::storage::EncodedStorage).

use super::{
    Cache, CloneShared,
    codec::{CloneFresh, CodecEngine},
    storage::EncodedStorage,
};
use crate::error::Result;

/// A cache backed by raw [`EncodedStorage`] and a [`CodecEngine`].
///
/// `EncodedCache` owns both a storage backend and a codec engine. During batch execution, each
/// worker gets its own cache handle: the store is cloned as a shared handle, while the codec is
/// cloned fresh with independent worker-local state.
///
/// The raw storage trait is sealed, so callers should construct this cache with crate-provided
/// storage backends such as `Fjall3Store`.
#[derive(Debug)]
pub struct EncodedCache<S, CE> {
    pub(crate) storage: S,
    engine: CE,
}

impl<S, CE> EncodedCache<S, CE> {
    /// Create an encoded cache from a raw store and a codec engine.
    pub fn new(storage: S, engine: CE) -> Self {
        Self { storage, engine }
    }
}

impl<S: CloneShared, CE: CloneFresh> CloneShared for EncodedCache<S, CE> {
    fn clone_shared(&self) -> Self {
        Self {
            storage: self.storage.clone_shared(),
            engine: self.engine.clone_fresh(),
        }
    }
}

impl<S, CE, T> Cache<T> for EncodedCache<S, CE>
where
    S: EncodedStorage,
    CE: CodecEngine<T>,
{
    fn fetch(&mut self, key: &[u8]) -> Result<Option<T>> {
        match self.storage.fetch_encoded(key)? {
            Some(raw) => match self.engine.decode(raw.as_ref()) {
                Ok(value) => Ok(Some(value)),
                Err(error) if error.is_cache_corruption() => {
                    warn!("[ssa-workflow] ignoring corrupted cache entry during read: {error}");
                    Ok(None)
                }
                Err(error) => Err(error.into()),
            },
            None => Ok(None),
        }
    }

    fn store(&mut self, key: &[u8], value: &T) -> Result<()> {
        match self.engine.encode(value) {
            Ok(encoded) => self.storage.store_encoded(key, encoded)?,
            Err(reason) => {
                warn!("[ssa-workflow] skipping cache write: {reason}");
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    mod fixtures {
        use std::{
            collections::HashMap,
            sync::atomic::{AtomicBool, Ordering},
        };

        use parking_lot::Mutex;

        use crate::{
            Result,
            cache::{
                codec::{CodecEngine, Error as CodecError, SkipReason},
                storage::{EncodedStorage, StorageResult},
            },
        };

        #[derive(Default)]
        pub(super) struct TestBytesStore {
            entries: Mutex<HashMap<Vec<u8>, Vec<u8>>>,
            write_called: AtomicBool,
        }

        impl TestBytesStore {
            pub(super) fn write_called(&self) -> bool {
                self.write_called.load(Ordering::Relaxed)
            }
        }

        impl EncodedStorage for TestBytesStore {
            type Encoded<'a>
                = Vec<u8>
            where
                Self: 'a;

            fn fetch_encoded(&self, key: &[u8]) -> StorageResult<Option<Self::Encoded<'_>>> {
                Ok(self.entries.lock().get(key).cloned())
            }

            fn store_encoded(&self, key: &[u8], encoded: &[u8]) -> StorageResult<()> {
                self.write_called.store(true, Ordering::Relaxed);
                self.entries.lock().insert(key.to_vec(), encoded.to_vec());
                Ok(())
            }
        }

        #[derive(Default)]
        pub(super) struct BytesEngine {
            encoded: Vec<u8>,
        }

        impl crate::cache::codec::CloneFresh for BytesEngine {
            fn clone_fresh(&self) -> Self {
                Self::default()
            }
        }

        impl CodecEngine<u32> for BytesEngine {
            const VALUE_FORMAT: crate::cache::codec::ValueFormat =
                crate::cache::codec::ValueFormat::new("test-bytes-u32-v1");

            fn encode(&mut self, value: &u32) -> std::result::Result<&[u8], SkipReason> {
                self.encoded.clear();
                self.encoded.extend_from_slice(&value.to_le_bytes());
                Ok(&self.encoded)
            }

            fn decode(&mut self, encoded: &[u8]) -> Result<u32, CodecError> {
                let bytes: [u8; 4] = encoded
                    .try_into()
                    .map_err(|_| CodecError::Fixture(FixtureError::Invalid("bad width")))?;
                Ok(u32::from_le_bytes(bytes))
            }
        }

        use crate::cache::codec::fixtures::Error as FixtureError;
    }

    mod cache_miss {
        use super::{super::EncodedCache, fixtures::*};
        use crate::{cache::Cache, error::Error};

        #[test]
        fn execute_error_does_not_store() {
            let store = TestBytesStore::default();
            let mut cache = EncodedCache::new(store, BytesEngine::default());

            let result = cache.fetch_or_execute(b"k", || Err(Error::Interrupted));

            assert!(matches!(result, Err(Error::Interrupted)));
            assert!(!cache.storage.write_called());
        }
    }
}
