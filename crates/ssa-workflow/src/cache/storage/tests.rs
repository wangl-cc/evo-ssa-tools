use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

use parking_lot::Mutex;

use super::CacheStore;
use crate::{
    cache::{
        Cache, CloneShared, EncodedCache,
        codec::{CheckedCodec, CodecEngine, SkipReason, fixtures::Error as FixtureError},
    },
    error::Result,
};

#[test]
fn test_default_store_skips_write_when_encode_is_rejected() -> Result<()> {
    struct SkipStore<'a> {
        write_called: &'a AtomicBool,
    }

    impl CacheStore for SkipStore<'_> {
        type Encoded<'a>
            = &'a [u8]
        where
            Self: 'a;

        fn fetch_encoded(&self, _: &[u8]) -> super::StorageResult<Option<Self::Encoded<'_>>> {
            Ok(None)
        }

        fn store_encoded(&self, _: &[u8], _: &[u8]) -> super::StorageResult<()> {
            self.write_called.store(true, Ordering::Relaxed);
            Ok(())
        }
    }

    #[derive(Default)]
    struct SkipEngine;

    impl crate::cache::codec::CloneFresh for SkipEngine {
        fn clone_fresh(&self) -> Self {
            Self
        }
    }

    impl CodecEngine<u32> for SkipEngine {
        const VALUE_FORMAT: crate::cache::codec::ValueFormat =
            crate::cache::codec::ValueFormat::new("test/skip-u32/v1");

        fn encode(&mut self, _: &u32) -> std::result::Result<&[u8], SkipReason> {
            Err(SkipReason::EncodedValueTooLarge {
                encoded_len: 123,
                max_len: 64,
            })
        }

        fn decode(&mut self, _: &[u8]) -> Result<u32, crate::cache::codec::Error> {
            unreachable!("skip store test never decodes")
        }
    }

    let write_called = AtomicBool::new(false);
    let store = SkipStore {
        write_called: &write_called,
    };
    let mut cache = EncodedCache::new(store, SkipEngine);

    cache.store(b"ignored", &42u32)?;
    assert!(!write_called.load(Ordering::Relaxed));
    Ok(())
}

#[test]
fn test_fetch_propagates_non_corruption_decode_error() {
    struct FailingStore;

    impl CacheStore for FailingStore {
        type Encoded<'a>
            = &'a [u8]
        where
            Self: 'a;

        fn fetch_encoded(&self, _: &[u8]) -> super::StorageResult<Option<Self::Encoded<'_>>> {
            Ok(Some(b"payload"))
        }

        fn store_encoded(&self, _: &[u8], _: &[u8]) -> super::StorageResult<()> {
            Ok(())
        }
    }

    #[derive(Default)]
    struct FailingDecodeEngine;

    impl crate::cache::codec::CloneFresh for FailingDecodeEngine {
        fn clone_fresh(&self) -> Self {
            Self
        }
    }

    impl CodecEngine<u32> for FailingDecodeEngine {
        const VALUE_FORMAT: crate::cache::codec::ValueFormat =
            crate::cache::codec::ValueFormat::new("test/failing-decode-u32/v1");

        fn encode(&mut self, _: &u32) -> std::result::Result<&[u8], SkipReason> {
            unreachable!("fetch test never encodes")
        }

        fn decode(&mut self, _: &[u8]) -> Result<u32, crate::cache::codec::Error> {
            Err(crate::cache::codec::Error::Fixture(FixtureError::Invalid(
                "decode failed",
            )))
        }
    }

    let store = FailingStore;
    let mut cache = EncodedCache::new(store, FailingDecodeEngine);
    let result = cache.fetch(b"ignored");

    assert!(matches!(
        result,
        Err(crate::error::Error::Codec(
            crate::cache::codec::Error::Fixture(FixtureError::Invalid("decode failed"))
        ))
    ));
}

#[test]
fn test_fetch_treats_checked_corruption_as_miss() -> Result<()> {
    #[derive(Default)]
    struct BytesEngine(Vec<u8>);

    impl crate::cache::codec::CloneFresh for BytesEngine {
        fn clone_fresh(&self) -> Self {
            Self::default()
        }
    }

    impl CodecEngine<u32> for BytesEngine {
        const VALUE_FORMAT: crate::cache::codec::ValueFormat =
            crate::cache::codec::ValueFormat::new("test/checked-bytes-u32/v1");

        fn encode(&mut self, value: &u32) -> std::result::Result<&[u8], SkipReason> {
            self.0.clear();
            self.0.extend_from_slice(&value.to_le_bytes());
            Ok(&self.0)
        }

        fn decode(&mut self, bytes: &[u8]) -> Result<u32, crate::cache::codec::Error> {
            let bytes: [u8; 4] = bytes.try_into().expect("u32 payload");
            Ok(u32::from_le_bytes(bytes))
        }
    }

    #[derive(Default)]
    struct SingleValueStore {
        value: Vec<u8>,
    }

    impl CacheStore for SingleValueStore {
        type Encoded<'a>
            = &'a [u8]
        where
            Self: 'a;

        fn fetch_encoded(&self, _: &[u8]) -> super::StorageResult<Option<Self::Encoded<'_>>> {
            Ok(Some(self.value.as_slice()))
        }

        fn store_encoded(&self, _: &[u8], _: &[u8]) -> super::StorageResult<()> {
            Ok(())
        }
    }

    let mut encode_engine = CheckedCodec::new(BytesEngine::default());
    let mut encoded = encode_engine
        .encode(&7u32)
        .expect("encoding should succeed")
        .to_vec();
    encoded[0] ^= 0x01;

    let store = SingleValueStore { value: encoded };
    let mut cache = EncodedCache::new(store, CheckedCodec::new(BytesEngine::default()));
    assert_eq!(cache.fetch(b"ignored")?, None);
    Ok(())
}

#[test]
fn test_fetch_or_execute_recomputes_when_checked_entry_is_corrupted() -> Result<()> {
    #[derive(Default)]
    struct BytesEngine(Vec<u8>);

    impl crate::cache::codec::CloneFresh for BytesEngine {
        fn clone_fresh(&self) -> Self {
            Self::default()
        }
    }

    impl CodecEngine<u32> for BytesEngine {
        const VALUE_FORMAT: crate::cache::codec::ValueFormat =
            crate::cache::codec::ValueFormat::new("test/corrupt-bytes-u32/v1");

        fn encode(&mut self, value: &u32) -> std::result::Result<&[u8], SkipReason> {
            self.0.clear();
            self.0.extend_from_slice(&value.to_le_bytes());
            Ok(&self.0)
        }

        fn decode(&mut self, bytes: &[u8]) -> Result<u32, crate::cache::codec::Error> {
            let bytes: [u8; 4] = bytes.try_into().expect("u32 payload");
            Ok(u32::from_le_bytes(bytes))
        }
    }

    struct TestStore {
        value: Vec<u8>,
        writes: Arc<AtomicUsize>,
    }

    impl CacheStore for TestStore {
        type Encoded<'a>
            = &'a [u8]
        where
            Self: 'a;

        fn fetch_encoded(&self, _: &[u8]) -> super::StorageResult<Option<Self::Encoded<'_>>> {
            Ok(Some(self.value.as_slice()))
        }

        fn store_encoded(&self, _: &[u8], _: &[u8]) -> super::StorageResult<()> {
            self.writes.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    let mut encode_engine = CheckedCodec::new(BytesEngine::default());
    let mut encoded = encode_engine
        .encode(&7u32)
        .expect("encoding should succeed")
        .to_vec();
    encoded.truncate(encoded.len() - 1);

    let writes = Arc::new(AtomicUsize::new(0));
    let store = TestStore {
        value: encoded,
        writes: writes.clone(),
    };
    let mut cache = EncodedCache::new(store, CheckedCodec::new(BytesEngine::default()));

    let value = cache.fetch_or_execute(b"ignored", || Ok(11u32))?;
    assert_eq!(value, 11);
    assert_eq!(writes.load(Ordering::Relaxed), 1);
    Ok(())
}

#[test]
fn test_fetch_or_execute_returns_cached_value_on_hit() -> Result<()> {
    #[derive(Default)]
    struct BytesEngine(Vec<u8>);

    impl crate::cache::codec::CloneFresh for BytesEngine {
        fn clone_fresh(&self) -> Self {
            Self::default()
        }
    }

    impl CodecEngine<u32> for BytesEngine {
        const VALUE_FORMAT: crate::cache::codec::ValueFormat =
            crate::cache::codec::ValueFormat::new("test/preloaded-bytes-u32/v1");

        fn encode(&mut self, value: &u32) -> std::result::Result<&[u8], SkipReason> {
            self.0.clear();
            self.0.extend_from_slice(&value.to_le_bytes());
            Ok(&self.0)
        }

        fn decode(&mut self, bytes: &[u8]) -> Result<u32, crate::cache::codec::Error> {
            let bytes: [u8; 4] = bytes.try_into().expect("u32 payload");
            Ok(u32::from_le_bytes(bytes))
        }
    }

    // Pre-encode a value to seed the store.
    let mut seed_engine = BytesEngine::default();
    let encoded_value = seed_engine.encode(&99u32).unwrap().to_vec();

    struct PreloadedStore {
        payload: Vec<u8>,
    }

    impl CacheStore for PreloadedStore {
        type Encoded<'a>
            = &'a [u8]
        where
            Self: 'a;

        fn fetch_encoded(&self, _: &[u8]) -> super::StorageResult<Option<Self::Encoded<'_>>> {
            Ok(Some(self.payload.as_slice()))
        }

        fn store_encoded(&self, _: &[u8], _: &[u8]) -> super::StorageResult<()> {
            Ok(())
        }
    }

    let store = PreloadedStore {
        payload: encoded_value,
    };
    let mut cache = EncodedCache::new(store, BytesEngine::default());
    let value = cache.fetch_or_execute(b"key", || Ok(0u32))?;
    assert_eq!(value, 99);
    Ok(())
}

#[test]
fn test_unit_store_clone_shared_is_noop() {
    let store = ();
    store.clone_shared();
}

// -- CacheStore + raw bytes store tests -------------------------------------
//
// These tests use an inline bytes store as a minimal storage-layer test double.

struct TestBytesStore {
    inner: Arc<Mutex<std::collections::HashMap<Vec<u8>, Vec<u8>>>>,
}

impl TestBytesStore {
    fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(std::collections::HashMap::new())),
        }
    }
}

impl super::CacheStore for TestBytesStore {
    type Encoded<'a>
        = Vec<u8>
    where
        Self: 'a;

    fn fetch_encoded(&self, key: &[u8]) -> super::StorageResult<Option<Self::Encoded<'_>>> {
        Ok(self.inner.lock().get(key).cloned())
    }

    fn store_encoded(&self, key: &[u8], encoded: &[u8]) -> super::StorageResult<()> {
        self.inner.lock().insert(key.to_owned(), encoded.to_vec());
        Ok(())
    }
}

#[test]
fn test_store_basic_roundtrip() -> crate::error::Result<()> {
    let store = TestBytesStore::new();
    let mut cache = EncodedCache::new(
        store,
        crate::cache::codec::fixtures::FixtureEngine::default(),
    );

    let value = 42u32;
    let key = b"test_sig";
    cache.store(key, &value)?;
    assert_eq!(cache.fetch(key)?, Some(value));
    assert_eq!(cache.fetch(b"non_existent")?, None::<u32>);
    assert!(Cache::<u64>::fetch(&mut cache, key).is_err());
    Ok(())
}
