use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use rand::Rng;
use ssa_workflow::{
    Compute,
    cache::{ManagedHashCache, codec::ValueFormat, storage::StorageNamespace},
    compute::{DependentStochasticInput, StochasticInput},
    error::Result,
    prelude::*,
};

#[test]
fn managed_hash_cache_reuses_results() -> Result<()> {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_clone = Arc::clone(&calls);
    let mut task = DeterministicTask::builder("double-v1")
        .function(move |input: usize| {
            calls_clone.fetch_add(1, Ordering::SeqCst);
            Ok(input * 2)
        })
        .cache(ManagedHashCache::<usize>::default())
        .build()?;

    assert_eq!(task.execute_one(5)?, 10);
    assert_eq!(task.execute_one(5)?, 10);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    Ok(())
}

#[test]
fn cloned_task_shares_bound_managed_hash_cache() -> Result<()> {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_clone = Arc::clone(&calls);

    let mut first = DeterministicTask::builder("shared-double-v1")
        .function(move |input: usize| {
            calls_clone.fetch_add(1, Ordering::SeqCst);
            Ok(input * 2)
        })
        .cache(ManagedHashCache::<usize>::default())
        .build()?;
    let mut second = first.clone();

    assert_eq!(first.execute_one(11)?, 22);
    assert_eq!(second.execute_one(11)?, 22);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    Ok(())
}

#[test]
fn independent_managed_hash_providers_do_not_share_space() -> Result<()> {
    let calls_a = Arc::new(AtomicUsize::new(0));
    let calls_b = Arc::new(AtomicUsize::new(0));

    let mut first = DeterministicTask::builder("same-id-v1")
        .function({
            let calls = Arc::clone(&calls_a);
            move |input: usize| {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(input * 2)
            }
        })
        .cache(ManagedHashCache::<usize>::default())
        .build()?;
    let mut second = DeterministicTask::builder("same-id-v1")
        .function({
            let calls = Arc::clone(&calls_b);
            move |input: usize| {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(input * 3)
            }
        })
        .cache(ManagedHashCache::<usize>::default())
        .build()?;

    assert_eq!(first.execute_one(11)?, 22);
    assert_eq!(second.execute_one(11)?, 33);
    assert_eq!(calls_a.load(Ordering::SeqCst), 1);
    assert_eq!(calls_b.load(Ordering::SeqCst), 1);
    Ok(())
}

#[test]
fn named_streams_affect_rng_but_not_cache_namespace() -> Result<()> {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_clone = Arc::clone(&calls);
    let mut task = StochasticTask::builder("computation-named-streams-v1")
        .streams(["waiting", "choice"])
        .function(move |rng, input: u64| {
            calls_clone.fetch_add(1, Ordering::SeqCst);
            let [waiting, choice] = rng.as_mut();
            Ok(input + (waiting.next_u64() % 10) + (choice.next_u64() % 10))
        })
        .cache(ManagedHashCache::<u64>::default())
        .build()?;

    let namespace = StorageNamespace::new(task.computation_path(), ValueFormat::new("memory-v1"));
    assert!(namespace.as_str().contains("computation-named-streams-v1"));
    assert!(!namespace.as_str().contains("waiting"));
    assert!(!namespace.as_str().contains("choice"));

    let input = StochasticInput::new(5u64, 0);
    let first = task.execute_one(input.clone())?;
    let second = task.execute_one(input)?;

    assert_eq!(first, second);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    Ok(())
}

#[test]
fn managed_transform_uses_child_path_for_output_space() -> Result<()> {
    let stage1_calls = Arc::new(AtomicUsize::new(0));
    let stage2_calls = Arc::new(AtomicUsize::new(0));
    let stage1_calls_clone = Arc::clone(&stage1_calls);
    let stage2_calls_clone = Arc::clone(&stage2_calls);

    let source = DeterministicTask::builder("double-v1")
        .function(move |input: usize| {
            stage1_calls_clone.fetch_add(1, Ordering::SeqCst);
            Ok(input * 2)
        })
        .cache(ManagedHashCache::<usize>::default())
        .build()?;

    let transform = source
        .transform("plus-one-v1")
        .function(move |intermediate: usize| {
            stage2_calls_clone.fetch_add(1, Ordering::SeqCst);
            Ok(intermediate + 1)
        })
        .cache(ManagedHashCache::<usize>::default())
        .build()?;

    let inputs: Vec<_> = (0..8usize).collect();
    let first = transform.with_inputs(inputs.clone()).collect()?;
    let second = transform.with_inputs(inputs).collect()?;

    assert_eq!(first, second);
    assert_eq!(first, (0..8usize).map(|i| i * 2 + 1).collect::<Vec<_>>());
    assert_eq!(stage1_calls.load(Ordering::SeqCst), 8);
    assert_eq!(stage2_calls.load(Ordering::SeqCst), 8);

    let namespace =
        StorageNamespace::new(transform.computation_path(), ValueFormat::new("memory-v1"));
    assert!(namespace.as_str().contains("plus-one-v1_double-v1"));
    Ok(())
}

#[test]
fn stochastic_transform_path_extends_source_path() -> Result<()> {
    let source = StochasticTask::builder("trajectory-v1")
        .function(|rng, ()| Ok(rand::Rng::next_u64(rng)))
        .cache(ManagedHashCache::<u64>::default())
        .build()?;
    let mut transform = source
        .stochastic_transform("resample-v1")
        .function(|rng, value: u64| Ok(value ^ rand::Rng::next_u64(rng)))
        .cache(ManagedHashCache::<u64>::default())
        .build()?;

    let namespace =
        StorageNamespace::new(transform.computation_path(), ValueFormat::new("memory-v1"));
    assert!(namespace.as_str().contains("resample-v1_trajectory-v1"));

    let input = DependentStochasticInput::from_source(StochasticInput::new((), 0), 0);
    assert_eq!(
        transform.execute_one(input.clone())?,
        transform.execute_one(input)?
    );
    Ok(())
}

#[cfg(feature = "lru")]
#[test]
fn managed_lru_cache_evicts_by_capacity() -> Result<()> {
    use std::num::NonZeroUsize;

    let calls = Arc::new(AtomicUsize::new(0));
    let calls_clone = Arc::clone(&calls);
    let mut task = DeterministicTask::builder("identity-v1")
        .function(move |input: usize| {
            calls_clone.fetch_add(1, Ordering::SeqCst);
            Ok(input)
        })
        .cache(ManagedLruCache::<usize>::new(
            NonZeroUsize::new(1).expect("capacity is non-zero"),
        ))
        .build()?;

    assert_eq!(task.execute_one(1)?, 1);
    assert_eq!(task.execute_one(2)?, 2);
    assert_eq!(task.execute_one(1)?, 1);
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    Ok(())
}

#[cfg(feature = "lru")]
#[test]
fn cloned_task_shares_managed_lru_eviction_state() -> Result<()> {
    use std::num::NonZeroUsize;

    let calls = Arc::new(AtomicUsize::new(0));
    let calls_clone = Arc::clone(&calls);
    let mut first = DeterministicTask::builder("shared-lru-v1")
        .function(move |input: usize| {
            calls_clone.fetch_add(1, Ordering::SeqCst);
            Ok(input)
        })
        .cache(ManagedLruCache::<usize>::new(
            NonZeroUsize::new(1).expect("capacity is non-zero"),
        ))
        .build()?;
    let mut second = first.clone();

    assert_eq!(first.execute_one(1)?, 1);
    assert_eq!(second.execute_one(2)?, 2);
    assert_eq!(first.execute_one(1)?, 1);
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    Ok(())
}

#[cfg(all(feature = "fjall3", feature = "bitcode06"))]
mod persistent_fjall3 {
    use ssa_workflow::{
        cache::{
            CanonicalEncode, PersistentCacheProvider, StorageProviderExt,
            codec::{Bitcode06, CheckedCodec, CodecEngine},
            storage::{EncodedStorage, Fjall3StorageProvider, Fjall3Store, StorageNamespace},
        },
        compute::{StochasticInput, StochasticTask},
    };

    use super::*;

    fn make_storage_provider()
    -> Result<(tempfile::TempDir, fjall3::Database, Fjall3StorageProvider)> {
        let dir = tempfile::tempdir().expect("tempdir should be created");
        let db = fjall3::Database::builder(&dir)
            .open()
            .map_err(ssa_workflow::cache::storage::StorageError::from)?;
        let provider = Fjall3StorageProvider::new(db.clone());
        Ok((dir, db, provider))
    }

    fn cache_provider(
        storage_provider: Fjall3StorageProvider,
    ) -> PersistentCacheProvider<Fjall3StorageProvider, Bitcode06> {
        storage_provider.with_codec(Bitcode06::default())
    }

    #[test]
    fn computation_change_uses_a_different_persistent_namespace() -> Result<()> {
        let (_dir, _db, storage_provider) = make_storage_provider()?;
        let provider = cache_provider(storage_provider);
        let calls_a = Arc::new(AtomicUsize::new(0));
        let calls_b = Arc::new(AtomicUsize::new(0));

        let mut task_a = StochasticTask::builder("computation-a-v1")
            .function({
                let calls = Arc::clone(&calls_a);
                move |rng, ()| {
                    calls.fetch_add(1, Ordering::SeqCst);
                    Ok(rand::Rng::next_u64(rng))
                }
            })
            .cache(provider.clone())
            .build()?;

        let mut task_b = StochasticTask::builder("computation-b-v1")
            .function({
                let calls = Arc::clone(&calls_b);
                move |rng, ()| {
                    calls.fetch_add(1, Ordering::SeqCst);
                    Ok(rand::Rng::next_u64(rng))
                }
            })
            .cache(provider)
            .build()?;

        let input = StochasticInput::new((), 0);
        let _ = task_a.execute_one(input.clone())?;
        let _ = task_b.execute_one(input.clone())?;
        let _ = task_a.execute_one(input.clone())?;
        let _ = task_b.execute_one(input)?;

        assert_eq!(calls_a.load(Ordering::SeqCst), 1);
        assert_eq!(calls_b.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[test]
    fn same_path_and_format_reopen_same_persistent_namespace() -> Result<()> {
        let (_dir, _db, storage_provider) = make_storage_provider()?;
        let provider = cache_provider(storage_provider);
        let calls = Arc::new(AtomicUsize::new(0));
        let input = StochasticInput::new((), 3);

        let mut first = StochasticTask::builder("reopen-computation-v1")
            .function({
                let calls = Arc::clone(&calls);
                move |_rng, ()| {
                    calls.fetch_add(1, Ordering::SeqCst);
                    Ok(11u64)
                }
            })
            .cache(provider.clone())
            .build()?;

        assert_eq!(first.execute_one(input.clone())?, 11);

        let mut second = StochasticTask::builder("reopen-computation-v1")
            .function({
                let calls = Arc::clone(&calls);
                move |_rng, ()| {
                    calls.fetch_add(1, Ordering::SeqCst);
                    Ok(22u64)
                }
            })
            .cache(provider)
            .build()?;

        assert_eq!(second.execute_one(input)?, 11);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[test]
    fn downstream_path_changes_with_upstream_computation() -> Result<()> {
        let (_dir, _db, storage_provider) = make_storage_provider()?;
        let provider = cache_provider(storage_provider);
        let source_calls = Arc::new(AtomicUsize::new(0));
        let transform_calls = Arc::new(AtomicUsize::new(0));

        let make_transform = |id| {
            let source_calls = Arc::clone(&source_calls);
            let transform_calls = Arc::clone(&transform_calls);
            StochasticTask::builder(id)
                .function(move |rng, ()| {
                    source_calls.fetch_add(1, Ordering::SeqCst);
                    Ok(rand::Rng::next_u64(rng))
                })
                .cache(provider.clone())
                .build()?
                .transform("as-string-v1")
                .function(move |value: u64| {
                    transform_calls.fetch_add(1, Ordering::SeqCst);
                    Ok(value.to_string())
                })
                .cache(provider.clone())
                .build()
        };

        let mut transform_a = make_transform("computation-a-v1")?;
        let mut transform_b = make_transform("computation-b-v1")?;
        let input = StochasticInput::new((), 7);

        let _ = transform_a.execute_one(input.clone())?;
        let _ = transform_b.execute_one(input.clone())?;
        let _ = transform_a.execute_one(input.clone())?;
        let _ = transform_b.execute_one(input)?;

        assert_eq!(source_calls.load(Ordering::SeqCst), 2);
        assert_eq!(transform_calls.load(Ordering::SeqCst), 2);
        Ok(())
    }

    #[test]
    fn value_format_changes_persistent_namespace() -> Result<()> {
        let (_dir, _db, storage_provider) = make_storage_provider()?;
        let calls = Arc::new(AtomicUsize::new(0));

        let make_plain_task = || {
            let calls = Arc::clone(&calls);
            StochasticTask::builder("computation-a-v1")
                .function(move |rng, ()| {
                    calls.fetch_add(1, Ordering::SeqCst);
                    Ok(rand::Rng::next_u64(rng))
                })
                .cache(storage_provider.clone().with_codec(Bitcode06::default()))
                .build()
        };
        let make_checked_task = || {
            let calls = Arc::clone(&calls);
            StochasticTask::builder("computation-a-v1")
                .function(move |rng, ()| {
                    calls.fetch_add(1, Ordering::SeqCst);
                    Ok(rand::Rng::next_u64(rng))
                })
                .cache(
                    storage_provider
                        .clone()
                        .with_codec(CheckedCodec::new(Bitcode06::default())),
                )
                .build()
        };

        let mut first = make_plain_task()?;
        let mut second = make_checked_task()?;
        let input = StochasticInput::new((), 1);

        let _ = first.execute_one(input.clone())?;
        let _ = second.execute_one(input)?;

        assert_eq!(calls.load(Ordering::SeqCst), 2);
        Ok(())
    }

    #[test]
    fn namespace_stores_raw_canonical_entry_keys() -> Result<()> {
        let (_dir, db, storage_provider) = make_storage_provider()?;
        let layout = <Bitcode06 as CodecEngine<u64>>::VALUE_FORMAT;
        let provider = cache_provider(storage_provider);

        let mut task = StochasticTask::builder("computation-a-v1")
            .function(|rng, ()| Ok(rand::Rng::next_u64(rng)))
            .cache(provider)
            .build()?;
        let input = StochasticInput::new((), 9);
        let _ = task.execute_one(input.clone())?;

        let namespace = StorageNamespace::new(task.computation_path(), layout);
        let store = Fjall3Store::open(db, namespace.as_str())?;
        let mut key_buffer = vec![0u8; StochasticInput::<()>::KEY_SIZE];
        let key = unsafe { input.encode_key_with_buffer(&mut key_buffer) };

        assert!(store.fetch_encoded(key)?.is_some());
        Ok(())
    }
}
