use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use rand::Rng;
use ssa_workflow::{
    cache::{Cache, CacheProvider, CloneShared},
    error::Result,
    identity::ComputationPath,
    prelude::*,
};

#[derive(Clone, Copy)]
struct RepetitionIndexProvider;

#[derive(Clone, Copy)]
struct RepetitionIndexCache;

impl CacheProvider<u64> for RepetitionIndexProvider {
    type Cache = RepetitionIndexCache;

    fn bind(self, _path: &ComputationPath) -> Result<Self::Cache> {
        Ok(RepetitionIndexCache)
    }
}

impl CloneShared for RepetitionIndexCache {
    fn clone_shared(&self) -> Self {
        *self
    }
}

impl Cache<u64> for RepetitionIndexCache {
    fn fetch(&mut self, key: &[u8]) -> Result<Option<u64>> {
        let mut repetition_bytes = [0u8; u64::SIZE];
        repetition_bytes.copy_from_slice(&key[key.len() - u64::SIZE..]);
        Ok(Some(u64::from_be_bytes(repetition_bytes)))
    }

    fn store(&mut self, _key: &[u8], _value: &u64) -> Result<()> {
        Ok(())
    }
}

#[test]
fn prelude_smoke_supports_execute_one_and_batch_collect() -> Result<()> {
    let call_count = Arc::new(AtomicUsize::new(0));
    let call_count_clone = Arc::clone(&call_count);

    let mut task = DeterministicTask::builder("prelude-smoke-v1")
        .function(move |input: usize| {
            call_count_clone.fetch_add(1, Ordering::SeqCst);
            Ok(input * 4)
        })
        .cache(ManagedHashCache::<usize>::default())
        .build()?;

    assert_eq!(task.execute_one(3usize)?, 12);

    let outputs: Vec<usize> = task.with_inputs(0..4usize).collect::<Result<Vec<_>>>()?;

    assert_eq!(outputs, vec![0, 4, 8, 12]);
    assert_eq!(call_count.load(Ordering::SeqCst), 4);
    Ok(())
}

#[test]
fn stochastic_repeated_input_collects_in_repetition_order() -> Result<()> {
    let task = StochasticTask::builder("repeated-input-order-v1")
        .function(|_, _: u32| Ok(u64::MAX))
        .cache(RepetitionIndexProvider)
        .build()?;

    let outputs: Vec<u64> = task
        .with_repeated_input(42u32, 5)
        .collect::<Result<Vec<_>>>()?;

    assert_eq!(outputs, vec![0, 1, 2, 3, 4]);
    Ok(())
}

#[test]
fn stochastic_repeated_input_zero_repetitions_is_empty() -> Result<()> {
    let call_count = Arc::new(AtomicUsize::new(0));
    let call_count_clone = Arc::clone(&call_count);

    let task = StochasticTask::builder("repeated-input-zero-v1")
        .function(move |rng, ()| {
            call_count_clone.fetch_add(1, Ordering::SeqCst);
            Ok(rng.next_u64())
        })
        .cache(ManagedHashCache::<u64>::default())
        .build()?;

    let outputs: Vec<u64> = task
        .with_repeated_input((), 0)
        .collect::<Result<Vec<_>>>()?;

    assert!(outputs.is_empty());
    assert_eq!(call_count.load(Ordering::SeqCst), 0);
    Ok(())
}

#[test]
fn stochastic_transform_reuses_cached_analysis() -> Result<()> {
    let experiment_calls = Arc::new(AtomicUsize::new(0));
    let experiment_calls_clone = Arc::clone(&experiment_calls);
    let analysis_calls = Arc::new(AtomicUsize::new(0));
    let analysis_calls_clone = Arc::clone(&analysis_calls);

    let transform = StochasticTask::builder("experiment-simple-v1")
        .function(move |rng, ()| {
            experiment_calls_clone.fetch_add(1, Ordering::SeqCst);
            Ok(rng.next_u64())
        })
        .cache(ManagedHashCache::<u64>::default())
        .build()?
        .transform("analysis-count-ones-v1")
        .function(move |sample: u64| {
            analysis_calls_clone.fetch_add(1, Ordering::SeqCst);
            Ok(sample.count_ones())
        })
        .cache(ManagedHashCache::<u32>::default())
        .build()?;

    let repetitions = 8;

    let results1: Vec<u32> = transform
        .with_repeated_input((), repetitions)
        .collect::<Result<Vec<_>>>()?;

    assert_eq!(results1.len(), repetitions);
    assert_eq!(experiment_calls.load(Ordering::SeqCst), repetitions);
    assert_eq!(analysis_calls.load(Ordering::SeqCst), repetitions);

    let results2: Vec<u32> = transform
        .with_repeated_input((), repetitions)
        .collect::<Result<Vec<_>>>()?;

    assert_eq!(results1, results2);
    assert_eq!(experiment_calls.load(Ordering::SeqCst), repetitions);
    assert_eq!(analysis_calls.load(Ordering::SeqCst), repetitions);
    Ok(())
}
