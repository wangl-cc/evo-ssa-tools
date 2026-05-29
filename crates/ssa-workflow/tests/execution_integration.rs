use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use rand::Rng;
use ssa_workflow::{error::Result, prelude::*};

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

    let outputs: Vec<usize> = task.with_inputs(0..4usize).collect()?;

    assert_eq!(outputs, vec![0, 4, 8, 12]);
    assert_eq!(call_count.load(Ordering::SeqCst), 4);
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

    let inputs: Vec<_> = (0..8u64)
        .map(|repetition| StochasticInput::new((), repetition))
        .collect();

    let results1: Vec<u32> = transform.with_inputs(inputs.clone()).collect()?;

    assert_eq!(results1.len(), inputs.len());
    assert_eq!(experiment_calls.load(Ordering::SeqCst), inputs.len());
    assert_eq!(analysis_calls.load(Ordering::SeqCst), inputs.len());

    let results2: Vec<u32> = transform.with_inputs(inputs).collect()?;

    assert_eq!(results1, results2);
    assert_eq!(experiment_calls.load(Ordering::SeqCst), 8);
    assert_eq!(analysis_calls.load(Ordering::SeqCst), 8);
    Ok(())
}
