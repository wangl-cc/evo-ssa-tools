use std::sync::atomic::{AtomicBool, Ordering};

use super::{CacheStore, WorkerForkStore};
use crate::{
    cache::codec::{
        CodecEngine, SkipReason, checked::CheckedCodec, fixtures::Error as FixtureError,
    },
    error::Result,
};

#[test]
fn test_unit_store_fetch_encoded_is_none() -> Result<()> {
    let store = ();
    assert!(store.fetch_encoded(b"ignored")?.is_none());
    Ok(())
}

#[test]
fn test_unit_store_skips_encode_in_no_cache_mode() -> Result<()> {
    #[derive(Default)]
    struct PanicOnEncode;

    impl CodecEngine<u32> for PanicOnEncode {
        fn encode(&mut self, _: &u32) -> std::result::Result<&[u8], SkipReason> {
            panic!("no-cache store should not encode values");
        }

        fn decode(&mut self, _: &[u8]) -> Result<u32, crate::cache::codec::Error> {
            unreachable!("no-cache fetch always misses")
        }
    }

    let store = ();
    let mut engine = PanicOnEncode;
    store.store::<u32, PanicOnEncode>(b"ignored", &mut engine, &42)?;
    Ok(())
}

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
fn test_unit_store_encoded_is_noop() -> Result<()> {
    let store = ();
    store.store_encoded(b"ignored", b"payload")?;
    Ok(())
}

#[test]
fn test_unit_store_fetch_or_execute_runs_execute() -> Result<()> {
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

    let store = ();
    let mut engine = BytesEngine::default();
    let value = store.fetch_or_execute(b"ignored", &mut engine, |_| Ok(7u32))?;
    assert_eq!(value, 7);
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
    use crate::cache::codec::compress::{CompressedCodec, algorithm::Lz4};

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

    let mut encode_engine = CompressedCodec::<crate::cache::codec::engine::bitcode::Bitcode, Lz4>::default();
    let mut encoded = encode_engine
        .encode(&42u64)
        .expect("encoding should succeed")
        .to_vec();
    // Flip a byte in the payload to corrupt the checksum
    encoded[0] ^= 0x01;

    let store = SingleValueStore { value: encoded };
    let mut read_engine = CompressedCodec::<crate::cache::codec::engine::bitcode::Bitcode, Lz4>::default();
    assert_eq!(store.fetch::<u64, _>(b"ignored", &mut read_engine)?, None);
    Ok(())
}

#[test]
fn test_unit_store_fork_is_noop() {
    let store = ();
    store.fork_store();
}
