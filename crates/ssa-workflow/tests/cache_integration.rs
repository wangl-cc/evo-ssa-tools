#![cfg(feature = "fjall3")]

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use ssa_workflow::{
    Compute,
    cache::{
        Cache, CacheProvider, CanonicalEncode, CloneShared, EncodedCache,
        codec::{CheckedCodec, CloneFresh, CodecEngine, Error as CodecError, SkipReason},
        storage::EncodedStorage,
    },
    compute::DeterministicTask,
    error::Result,
    identity::ComputationPath,
};

struct SingleCacheProvider<C>(C);

impl<C: CloneShared> Clone for SingleCacheProvider<C> {
    fn clone(&self) -> Self {
        Self(self.0.clone_shared())
    }
}

impl<T, C> CacheProvider<T> for SingleCacheProvider<C>
where
    C: Cache<T> + CloneShared,
{
    type Cache = C;

    fn bind(self, _path: &ComputationPath) -> Result<Self::Cache> {
        Ok(self.0.clone_shared())
    }
}

fn fjall_store(
    name: &str,
) -> Result<(tempfile::TempDir, ssa_workflow::cache::storage::Fjall3Store)> {
    let dir = tempfile::tempdir().expect("tempdir should be created");
    let db = fjall3::Database::builder(&dir)
        .open()
        .map_err(ssa_workflow::cache::storage::StorageError::from)?;
    let store = ssa_workflow::cache::storage::Fjall3Store::open(db, name)?;
    Ok((dir, store))
}

struct TaggedUsizeEngine {
    tag: u8,
    buffer: Vec<u8>,
}

impl TaggedUsizeEngine {
    fn new(tag: u8) -> Self {
        Self {
            tag,
            buffer: Vec::new(),
        }
    }
}

impl CloneFresh for TaggedUsizeEngine {
    fn clone_fresh(&self) -> Self {
        Self::new(self.tag)
    }
}

impl CodecEngine<usize> for TaggedUsizeEngine {
    const VALUE_FORMAT: ssa_workflow::cache::codec::ValueFormat =
        ssa_workflow::cache::codec::ValueFormat::new("test-tagged-usize-v1");

    fn encode(&mut self, value: &usize) -> std::result::Result<&[u8], SkipReason> {
        self.buffer.clear();
        self.buffer.push(self.tag);
        self.buffer.extend_from_slice(&value.to_le_bytes());
        Ok(&self.buffer)
    }

    fn decode(&mut self, bytes: &[u8]) -> Result<usize, CodecError> {
        let Some((&tag, payload)) = bytes.split_first() else {
            panic!("expected tagged payload");
        };
        assert_eq!(tag, self.tag);
        let value_bytes: [u8; core::mem::size_of::<usize>()] =
            payload.try_into().expect("payload should store one usize");
        Ok(usize::from_le_bytes(value_bytes))
    }
}

#[derive(Default)]
struct BytesUsizeEngine(Vec<u8>);

impl ssa_workflow::cache::codec::CloneFresh for BytesUsizeEngine {
    fn clone_fresh(&self) -> Self {
        Self::default()
    }
}

impl CodecEngine<usize> for BytesUsizeEngine {
    const VALUE_FORMAT: ssa_workflow::cache::codec::ValueFormat =
        ssa_workflow::cache::codec::ValueFormat::new("test-bytes-usize-v1");

    fn encode(&mut self, value: &usize) -> std::result::Result<&[u8], SkipReason> {
        self.0.clear();
        self.0.extend_from_slice(&value.to_le_bytes());
        Ok(&self.0)
    }

    fn decode(&mut self, bytes: &[u8]) -> Result<usize, CodecError> {
        let value_bytes: [u8; core::mem::size_of::<usize>()] =
            bytes.try_into().expect("payload should store one usize");
        Ok(usize::from_le_bytes(value_bytes))
    }
}

#[test]
fn encoded_cache_uses_configured_fresh_codec_in_parallel_execution() -> Result<()> {
    let call_count = Arc::new(AtomicUsize::new(0));
    let call_count_clone = Arc::clone(&call_count);
    let (_dir, store) = fjall_store("encoded-parallel")?;

    let compute = DeterministicTask::builder("test-encoded-parallel-v1")
        .function(move |i: usize| {
            call_count_clone.fetch_add(1, Ordering::SeqCst);
            Ok(i + 10)
        })
        .cache(SingleCacheProvider(EncodedCache::new(
            store,
            TaggedUsizeEngine::new(0xA5),
        )))
        .build()?;

    let outputs1 = compute.with_inputs(0..8usize).collect::<Result<Vec<_>>>()?;
    let outputs2 = compute.with_inputs(0..8usize).collect::<Result<Vec<_>>>()?;

    let expected: Vec<_> = (0..8usize).map(|i| i + 10).collect();
    assert_eq!(outputs1, expected);
    assert_eq!(outputs2, expected);
    assert_eq!(call_count.load(Ordering::SeqCst), 8);
    Ok(())
}

#[test]
fn encoded_cache_treats_checked_corruption_as_miss_in_execution_flow() -> Result<()> {
    let compute_calls = Arc::new(AtomicUsize::new(0));
    let (_dir, store) = fjall_store("checked-corruption")?;
    let store_handle = store.clone_shared();

    let mut compute = DeterministicTask::builder("test-checked-corruption-v1")
        .function({
            let compute_calls = Arc::clone(&compute_calls);
            move |i: usize| {
                compute_calls.fetch_add(1, Ordering::SeqCst);
                Ok(i * 10)
            }
        })
        .cache(SingleCacheProvider(EncodedCache::new(
            store,
            CheckedCodec::new(BytesUsizeEngine::default()),
        )))
        .build()?;

    assert_eq!(compute.execute_one(3usize)?, 30);
    assert_eq!(compute_calls.load(Ordering::SeqCst), 1);

    let mut key_buffer = vec![0u8; usize::SIZE];
    let key = unsafe { 3usize.encode_with_buffer(&mut key_buffer) }.to_vec();
    let mut corrupted = store_handle
        .fetch_encoded(&key)?
        .expect("value should have been stored")
        .as_ref()
        .to_vec();
    corrupted[0] ^= 0x01;
    store_handle.store_encoded(&key, &corrupted)?;

    assert_eq!(compute.execute_one(3usize)?, 30);
    assert_eq!(compute_calls.load(Ordering::SeqCst), 2);
    Ok(())
}
