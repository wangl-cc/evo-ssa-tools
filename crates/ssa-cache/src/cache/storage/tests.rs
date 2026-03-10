use std::sync::atomic::{AtomicBool, Ordering};

use super::{CacheStore, WorkerForkStore};
use crate::{
    cache::codec::{CodecEngine, SkipReason},
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
fn test_unit_store_fork_is_noop() {
    let store = ();
    store.fork_store();
}
