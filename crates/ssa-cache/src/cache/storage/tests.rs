use super::CacheStore;
use crate::{
    cache::codec::{CodecEngine, SkipReason},
    error::Result,
};

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
fn test_unit_store_encoded_is_noop() -> Result<()> {
    let store = ();
    store.store_encoded(b"ignored", b"payload")?;
    Ok(())
}
