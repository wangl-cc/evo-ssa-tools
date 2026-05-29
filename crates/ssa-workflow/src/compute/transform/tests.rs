use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    thread::sleep,
    time::Duration,
};

use super::*;
use crate::{Result, compute::execution::Compute, prelude::*};

#[test]
fn test_transform_two_stage_caching() -> Result<()> {
    let stage1_calls = Arc::new(AtomicUsize::new(0));
    let stage1_calls_clone = stage1_calls.clone();
    let stage2_calls = Arc::new(AtomicUsize::new(0));
    let stage2_calls_clone = stage2_calls.clone();

    let transform = DeterministicTask::builder("test-two-stage-source-v1")
        .function(move |input| {
            stage1_calls_clone.fetch_add(1, Ordering::SeqCst);
            sleep(Duration::from_millis(10));
            Ok(input * 2)
        })
        .cache(ManagedHashCache::<usize>::default())
        .build()?
        .transform("test-two-stage-plus-ten-v1")
        .function(move |intermediate| {
            stage2_calls_clone.fetch_add(1, Ordering::SeqCst);
            sleep(Duration::from_millis(10));
            Ok(intermediate + 10)
        })
        .cache(ManagedHashCache::<usize>::default())
        .build()?;

    let results1 = transform.with_inputs(0..5usize).collect()?;

    let expected: Vec<usize> = (0..5).map(|i| i * 2 + 10).collect();
    assert_eq!(results1, expected);
    assert_eq!(stage1_calls.load(Ordering::SeqCst), 5);
    assert_eq!(stage2_calls.load(Ordering::SeqCst), 5);

    let results2 = transform.with_inputs(0..5).collect()?;

    assert_eq!(results2, expected);
    assert_eq!(stage1_calls.load(Ordering::SeqCst), 5);
    assert_eq!(stage2_calls.load(Ordering::SeqCst), 5);

    Ok(())
}

#[test]
fn test_transform_source_and_transform_cache_split() -> Result<()> {
    let stage1_calls = Arc::new(AtomicUsize::new(0));
    let stage2_calls = Arc::new(AtomicUsize::new(0));
    let stage1 = DeterministicTask::builder("test-cache-split-source-v1")
        .function({
            let stage1_calls = stage1_calls.clone();
            move |input: usize| {
                stage1_calls.fetch_add(1, Ordering::SeqCst);
                sleep(Duration::from_millis(10));
                Ok(input * 2)
            }
        })
        .cache(ManagedHashCache::<usize>::default())
        .build()?;

    let transform1 = {
        let stage2_calls = stage2_calls.clone();
        stage1
            .clone()
            .transform("test-cache-split-format-a-v1")
            .function(move |intermediate| {
                stage2_calls.fetch_add(1, Ordering::SeqCst);
                sleep(Duration::from_millis(10));
                Ok(format!("Result: {}", intermediate))
            })
            .cache(ManagedHashCache::<String>::default())
            .build()?
    };

    let results1 = transform1.with_inputs(0..3).collect()?;

    let expected: Vec<String> = (0..3).map(|i| format!("Result: {}", i * 2)).collect();
    assert_eq!(results1, expected);
    assert_eq!(stage1_calls.load(Ordering::SeqCst), 3);
    assert_eq!(stage2_calls.load(Ordering::SeqCst), 3);

    let transform2 = {
        let stage2_calls = stage2_calls.clone();
        stage1
            .transform("test-cache-split-format-b-v1")
            .function(move |intermediate| {
                stage2_calls.fetch_add(1, Ordering::SeqCst);
                sleep(Duration::from_millis(10));
                Ok(format!("Result: {}", intermediate))
            })
            .cache(ManagedHashCache::<String>::default())
            .build()?
    };

    let results2 = transform2.with_inputs(0..3).collect()?;

    assert_eq!(results2, expected);
    assert_eq!(stage1_calls.load(Ordering::SeqCst), 3);
    assert_eq!(stage2_calls.load(Ordering::SeqCst), 6);

    Ok(())
}

#[test]
fn test_transform_with_different_types() -> Result<()> {
    let transform = DeterministicTask::builder("test-different-types-source-v1")
        .function(|input: u32| Ok(input as u64 * 100))
        .cache(ManagedHashCache::<u64>::default())
        .build()?
        .transform("test-different-types-string-v1")
        .function(|intermediate: u64| Ok(format!("Value: {}", intermediate)))
        .cache(ManagedHashCache::<String>::default())
        .build()?;

    let results = transform.with_inputs(0..5u32).collect()?;

    let expected: Vec<String> = (0..5u32).map(|i| format!("Value: {}", i * 100)).collect();
    assert_eq!(results, expected);

    Ok(())
}

#[test]
fn dependent_input_encodes_param_before_source() {
    let input = DependentInput::new(0x0102u16, 0x0304u16);
    let mut buffer = vec![0u8; DependentInput::<u16, u16>::SIZE];
    let encoded = unsafe { input.encode_with_buffer(&mut buffer) };

    assert_eq!(encoded, &[0x01, 0x02, 0x03, 0x04]);
}

#[test]
fn dependent_stochastic_input_encodes_param_source_then_repetition() {
    let input = DependentStochasticInput::new(0x0102u16, 0x0304u16, 5);
    let mut buffer = vec![0u8; DependentStochasticInput::<u16, u16>::SIZE];
    let encoded = unsafe { input.encode_with_buffer(&mut buffer) };

    assert_eq!(encoded, &[0x01, 0x02, 0x03, 0x04, 0, 0, 0, 0, 0, 0, 0, 5]);
}

#[test]
fn parameterized_transform_recomputes_only_transform_for_param_changes() -> Result<()> {
    let source_calls = Arc::new(AtomicUsize::new(0));
    let transform_calls = Arc::new(AtomicUsize::new(0));

    let source = DeterministicTask::builder("source-value-v1")
        .function({
            let source_calls = Arc::clone(&source_calls);
            move |input: u16| {
                source_calls.fetch_add(1, Ordering::SeqCst);
                Ok(input * 2)
            }
        })
        .cache(ManagedHashCache::<u16>::default());

    let mut transform = source
        .build()?
        .transform("analysis-offset-v1")
        .function_with_param({
            let transform_calls = Arc::clone(&transform_calls);
            move |value, offset: u16| {
                transform_calls.fetch_add(1, Ordering::SeqCst);
                Ok(value + offset)
            }
        })
        .cache(ManagedHashCache::<u16>::default())
        .build()?;

    assert_eq!(transform.execute_one(DependentInput::new(1, 10))?, 21);
    assert_eq!(transform.execute_one(DependentInput::new(2, 10))?, 22);
    assert_eq!(transform.execute_one(DependentInput::new(1, 10))?, 21);
    assert_eq!(source_calls.load(Ordering::SeqCst), 1);
    assert_eq!(transform_calls.load(Ordering::SeqCst), 2);
    Ok(())
}

#[test]
fn cloned_parameterized_transform_shares_cache() -> Result<()> {
    let transform_calls = Arc::new(AtomicUsize::new(0));

    let source = DeterministicTask::builder("source-value-v1")
        .function(|input: u16| Ok(input * 2))
        .cache(ManagedHashCache::<u16>::default())
        .build()?;

    let mut transform = source
        .transform("analysis-offset-v1")
        .function_with_param({
            let transform_calls = Arc::clone(&transform_calls);
            move |value, offset: u16| {
                transform_calls.fetch_add(1, Ordering::SeqCst);
                Ok(value + offset)
            }
        })
        .cache(ManagedHashCache::<u16>::default())
        .build()?;
    let mut cloned = transform.clone();

    let input = DependentInput::new(1, 10);
    assert_eq!(transform.execute_one(input.clone())?, 21);
    assert_eq!(cloned.execute_one(input)?, 21);
    assert_eq!(transform_calls.load(Ordering::SeqCst), 1);
    Ok(())
}

#[test]
fn stochastic_transform_repetition_changes_output_and_hits_cache() -> Result<()> {
    let source_calls = Arc::new(AtomicUsize::new(0));
    let transform_calls = Arc::new(AtomicUsize::new(0));

    let source = DeterministicTask::builder("source-value-v1")
        .function({
            let source_calls = Arc::clone(&source_calls);
            move |input: u16| {
                source_calls.fetch_add(1, Ordering::SeqCst);
                Ok(input * 2)
            }
        })
        .cache(ManagedHashCache::<u16>::default());

    let mut transform = source
        .build()?
        .stochastic_transform("analysis-sample-v1")
        .function_with_param({
            let transform_calls = Arc::clone(&transform_calls);
            move |rng, value, offset: u16| {
                transform_calls.fetch_add(1, Ordering::SeqCst);
                Ok(value as u64 + offset as u64 + rand::Rng::next_u64(rng))
            }
        })
        .cache(ManagedHashCache::<u64>::default())
        .build()?;

    let input0 = DependentStochasticInput::new(1, 10, 0);
    let input1 = DependentStochasticInput::new(1, 10, 1);
    let first = transform.execute_one(input0.clone())?;
    let second = transform.execute_one(input1)?;
    let first_again = transform.execute_one(input0)?;

    assert_ne!(first, second);
    assert_eq!(first, first_again);
    assert_eq!(source_calls.load(Ordering::SeqCst), 1);
    assert_eq!(transform_calls.load(Ordering::SeqCst), 2);
    Ok(())
}

#[test]
fn cloned_stochastic_transform_shares_cache() -> Result<()> {
    let transform_calls = Arc::new(AtomicUsize::new(0));

    let source = DeterministicTask::builder("source-value-v1")
        .function(|input: u16| Ok(input * 2))
        .cache(ManagedHashCache::<u16>::default())
        .build()?;

    let mut transform = source
        .stochastic_transform("analysis-resample-v1")
        .function({
            let transform_calls = Arc::clone(&transform_calls);
            move |rng, value| {
                transform_calls.fetch_add(1, Ordering::SeqCst);
                Ok(value as u64 + rand::Rng::next_u64(rng))
            }
        })
        .cache(ManagedHashCache::<u64>::default())
        .build()?;
    let mut cloned = transform.clone();

    let input = DependentStochasticInput::from_source(10, 0);
    let first = transform.execute_one(input.clone())?;
    let second = cloned.execute_one(input)?;
    assert_eq!(first, second);
    assert_eq!(transform_calls.load(Ordering::SeqCst), 1);
    Ok(())
}

#[test]
fn cloned_parameterized_stochastic_transform_shares_cache() -> Result<()> {
    let transform_calls = Arc::new(AtomicUsize::new(0));

    let source = DeterministicTask::builder("param-stochastic-source-v1")
        .function(|input: u16| Ok(input * 2))
        .cache(ManagedHashCache::<u16>::default())
        .build()?;

    let mut transform = source
        .stochastic_transform("param-stochastic-resample-v1")
        .function_with_param({
            let transform_calls = Arc::clone(&transform_calls);
            move |rng, value, offset: u16| {
                transform_calls.fetch_add(1, Ordering::SeqCst);
                Ok(value as u64 + offset as u64 + rand::Rng::next_u64(rng))
            }
        })
        .cache(ManagedHashCache::<u64>::default())
        .build()?;
    let mut cloned = transform.clone();

    let input = DependentStochasticInput::new(7u16, 10u16, 0);
    let first = transform.execute_one(input.clone())?;
    let second = cloned.execute_one(input)?;

    assert_eq!(first, second);
    assert_eq!(transform_calls.load(Ordering::SeqCst), 1);
    Ok(())
}

#[test]
fn stochastic_transform_without_param_uses_source_and_repetition() -> Result<()> {
    let source_calls = Arc::new(AtomicUsize::new(0));
    let transform_calls = Arc::new(AtomicUsize::new(0));

    let source = DeterministicTask::builder("source-value-v1")
        .function({
            let source_calls = Arc::clone(&source_calls);
            move |input: u16| {
                source_calls.fetch_add(1, Ordering::SeqCst);
                Ok(input * 2)
            }
        })
        .cache(ManagedHashCache::<u16>::default());

    let mut transform = source
        .build()?
        .stochastic_transform("analysis-resample-v1")
        .function({
            let transform_calls = Arc::clone(&transform_calls);
            move |rng, value| {
                transform_calls.fetch_add(1, Ordering::SeqCst);
                Ok(value as u64 + rand::Rng::next_u64(rng))
            }
        })
        .cache(ManagedHashCache::<u64>::default())
        .build()?;

    let input0 = DependentStochasticInput::from_source(10, 0);
    let input1 = DependentStochasticInput::from_source(10, 1);
    let first = transform.execute_one(input0.clone())?;
    let second = transform.execute_one(input1)?;
    let first_again = transform.execute_one(input0)?;

    assert_ne!(first, second);
    assert_eq!(first, first_again);
    assert_eq!(source_calls.load(Ordering::SeqCst), 1);
    assert_eq!(transform_calls.load(Ordering::SeqCst), 2);
    Ok(())
}

#[test]
fn named_stochastic_transform_uses_distinct_streams_and_hits_cache() -> Result<()> {
    const LEFT: RandomVariable = RandomVariable::new("analysis-left-v1");
    const RIGHT: RandomVariable = RandomVariable::new("analysis-right-v1");

    let transform_calls = Arc::new(AtomicUsize::new(0));
    let source = DeterministicTask::builder("source-value-v1")
        .function(|input: u16| Ok(input * 2))
        .cache(ManagedHashCache::<u16>::default())
        .build()?;

    let mut transform = source
        .stochastic_transform("analysis-named-sample-v1")
        .streams([LEFT, RIGHT])
        .function_with_param({
            let transform_calls = Arc::clone(&transform_calls);
            move |rngs, value, offset: u16| {
                transform_calls.fetch_add(1, Ordering::SeqCst);
                let [left, right] = rngs.as_mut();
                Ok((value as u64)
                    .wrapping_add(offset as u64)
                    .wrapping_add(rand::Rng::next_u64(left))
                    .wrapping_add(rand::Rng::next_u64(right)))
            }
        })
        .cache(ManagedHashCache::<u64>::default())
        .build()?;

    let input0 = DependentStochasticInput::new(1, 10, 0);
    let input1 = DependentStochasticInput::new(1, 10, 1);
    let first = transform.execute_one(input0.clone())?;
    let second = transform.execute_one(input1)?;
    let first_again = transform.execute_one(input0)?;

    assert_ne!(first, second);
    assert_eq!(first, first_again);
    assert_eq!(transform_calls.load(Ordering::SeqCst), 2);
    Ok(())
}

#[test]
fn cloned_named_stochastic_transform_shares_cache() -> Result<()> {
    const LEFT: RandomVariable = RandomVariable::new("analysis-left-v1");
    const RIGHT: RandomVariable = RandomVariable::new("analysis-right-v1");

    let transform_calls = Arc::new(AtomicUsize::new(0));
    let source = DeterministicTask::builder("source-value-v1")
        .function(|input: u16| Ok(input * 2))
        .cache(ManagedHashCache::<u16>::default())
        .build()?;

    let mut transform = source
        .stochastic_transform("analysis-named-resample-v1")
        .streams([LEFT, RIGHT])
        .function({
            let transform_calls = Arc::clone(&transform_calls);
            move |rngs, value| {
                transform_calls.fetch_add(1, Ordering::SeqCst);
                let [left, right] = rngs.as_mut();
                Ok((value as u64)
                    .wrapping_add(rand::Rng::next_u64(left))
                    .wrapping_add(rand::Rng::next_u64(right)))
            }
        })
        .cache(ManagedHashCache::<u64>::default())
        .build()?;
    let mut cloned = transform.clone();

    let input = DependentStochasticInput::from_source(10, 0);
    let first = transform.execute_one(input.clone())?;
    let second = cloned.execute_one(input)?;
    assert_eq!(first, second);
    assert_eq!(transform_calls.load(Ordering::SeqCst), 1);
    Ok(())
}

#[test]
fn named_stochastic_transform_without_param_uses_distinct_streams() -> Result<()> {
    const LEFT: RandomVariable = RandomVariable::new("analysis-left-v1");
    const RIGHT: RandomVariable = RandomVariable::new("analysis-right-v1");

    let transform_calls = Arc::new(AtomicUsize::new(0));
    let source = DeterministicTask::builder("source-value-v1")
        .function(|input: u16| Ok(input * 2))
        .cache(ManagedHashCache::<u16>::default())
        .build()?;

    let mut transform = source
        .stochastic_transform("analysis-named-resample-v1")
        .streams([LEFT, RIGHT])
        .function({
            let transform_calls = Arc::clone(&transform_calls);
            move |rngs, value| {
                transform_calls.fetch_add(1, Ordering::SeqCst);
                let [left, right] = rngs.as_mut();
                Ok((value as u64)
                    .wrapping_add(rand::Rng::next_u64(left))
                    .wrapping_add(rand::Rng::next_u64(right)))
            }
        })
        .cache(ManagedHashCache::<u64>::default())
        .build()?;

    let input0 = DependentStochasticInput::from_source(10, 0);
    let input1 = DependentStochasticInput::from_source(10, 1);
    let first = transform.execute_one(input0.clone())?;
    let second = transform.execute_one(input1)?;
    let first_again = transform.execute_one(input0)?;

    assert_ne!(first, second);
    assert_eq!(first, first_again);
    assert_eq!(transform_calls.load(Ordering::SeqCst), 2);
    Ok(())
}
