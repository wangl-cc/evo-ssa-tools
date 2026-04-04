use std::sync::atomic::{AtomicBool, Ordering};

use super::CacheStore;
use crate::{
    cache::{
        CloneShared,
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

    impl CodecEngine<u32> for SkipEngine {
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
    let mut engine = SkipEngine;

    store.store(b"ignored", &mut engine, &42u32)?;
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

    impl CodecEngine<u32> for FailingDecodeEngine {
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
    let mut engine = FailingDecodeEngine;
    let result = store.fetch::<u32, _>(b"ignored", &mut engine);

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

    impl CodecEngine<u32> for BytesEngine {
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
    let mut read_engine = CheckedCodec::new(BytesEngine::default());
    assert_eq!(store.fetch(b"ignored", &mut read_engine)?, None);
    Ok(())
}

#[test]
fn test_fetch_or_execute_recomputes_when_checked_entry_is_corrupted() -> Result<()> {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    #[derive(Default)]
    struct BytesEngine(Vec<u8>);

    impl CodecEngine<u32> for BytesEngine {
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
    let mut engine = CheckedCodec::new(BytesEngine::default());

    let value = store.fetch_or_execute(b"ignored", &mut engine, |_| Ok(11u32))?;
    assert_eq!(value, 11);
    assert_eq!(writes.load(Ordering::Relaxed), 1);
    Ok(())
}

#[cfg(feature = "lz4")]
#[test]
fn test_fetch_treats_compress_corruption_as_miss() -> Result<()> {
    use crate::cache::codec::compress::{CompressedCodec, Lz4};

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

    let mut encode_engine = CompressedCodec::<crate::cache::codec::Bitcode06, Lz4>::default();
    let mut encoded = encode_engine
        .encode(&42u64)
        .expect("encoding should succeed")
        .to_vec();
    // Flip a byte in the payload to corrupt the checksum
    encoded[0] ^= 0x01;

    let store = SingleValueStore { value: encoded };
    let mut read_engine = CompressedCodec::<crate::cache::codec::Bitcode06, Lz4>::default();
    assert_eq!(store.fetch::<u64, _>(b"ignored", &mut read_engine)?, None);
    Ok(())
}

#[test]
fn test_fetch_or_execute_returns_cached_value_on_hit() -> Result<()> {
    #[derive(Default)]
    struct BytesEngine(Vec<u8>);

    impl CodecEngine<u32> for BytesEngine {
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
    let mut engine = BytesEngine::default();
    let value = store.fetch_or_execute(b"key", &mut engine, |_| Ok(0u32))?;
    assert_eq!(value, 99);
    Ok(())
}

#[test]
fn test_unit_store_clone_shared_is_noop() {
    let store = ();
    store.clone_shared();
}

// -- CacheStore + compressed codec tests ------------------------------------
//
// These tests use an inline shared-bytes store so that raw encoded bytes can
// be inspected after a store/fetch roundtrip.

use std::sync::{Arc, Mutex};

struct SharedBytesStore {
    inner: Arc<Mutex<std::collections::HashMap<Vec<u8>, Vec<u8>>>>,
}

impl SharedBytesStore {
    fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(std::collections::HashMap::new())),
        }
    }
}

impl super::CacheStore for SharedBytesStore {
    type Encoded<'a>
        = Vec<u8>
    where
        Self: 'a;

    fn fetch_encoded(&self, key: &[u8]) -> super::StorageResult<Option<Self::Encoded<'_>>> {
        Ok(self.inner.lock().unwrap().get(key).cloned())
    }

    fn store_encoded(&self, key: &[u8], encoded: &[u8]) -> super::StorageResult<()> {
        self.inner
            .lock()
            .unwrap()
            .insert(key.to_owned(), encoded.to_vec());
        Ok(())
    }
}

#[test]
fn test_store_basic_roundtrip() -> crate::error::Result<()> {
    let store = SharedBytesStore::new();
    let mut engine = crate::cache::codec::fixtures::FixtureEngine::default();

    let value = 42u32;
    let key = b"test_sig";
    store.store::<u32, _>(key, &mut engine, &value)?;
    assert_eq!(store.fetch::<u32, _>(key, &mut engine)?, Some(value));
    assert_eq!(store.fetch::<u32, _>(b"non_existent", &mut engine)?, None);
    assert!(store.fetch::<u64, _>(key, &mut engine).is_err());
    Ok(())
}

#[cfg(feature = "lz4")]
#[test]
fn test_compressed_value_frame_layout() -> crate::error::Result<()> {
    use crate::cache::codec::{
        compress::{CompressedCodec, Lz4},
        fixtures::FixtureEngine,
    };

    type Lz4Engine = CompressedCodec<FixtureEngine, Lz4>;

    let store = SharedBytesStore::new();
    let mut engine = Lz4Engine::default();
    let key = b"compressible";
    let value = "a".repeat(96 * 1024);

    store.store::<String, Lz4Engine>(key, &mut engine, &value)?;
    assert_eq!(
        store.fetch::<String, Lz4Engine>(key, &mut engine)?,
        Some(value)
    );

    let encoded = store.get_raw(key).expect("value should be stored");
    assert!(!encoded.is_empty());
    assert_eq!(encoded[0] & 0b1111_0000, 0b0001_0000);
    assert_eq!(encoded[0] & 0b0000_1111, 0b0001);
    Ok(())
}

#[cfg(all(feature = "lz4", feature = "bitcode06"))]
#[test]
fn test_compressed_value_skips_oversize() -> crate::error::Result<()> {
    use crate::cache::codec::compress::{CompressedCodec, Lz4, fixtures::SizedBytesEngine};

    type Lz4Engine = CompressedCodec<SizedBytesEngine, Lz4>;

    let store = SharedBytesStore::new();
    let mut engine = Lz4Engine::default().with_max_len(64 * 1024 * 1024);
    let key = b"oversize";
    let value = 64 * 1024 * 1024 + 1;

    store.store::<usize, Lz4Engine>(key, &mut engine, &value)?;
    assert_eq!(store.fetch::<usize, Lz4Engine>(key, &mut engine)?, None);
    Ok(())
}
