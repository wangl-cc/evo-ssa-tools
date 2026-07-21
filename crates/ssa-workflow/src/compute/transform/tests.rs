use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use super::*;
use crate::{Result, compute::execution::Compute, prelude::*};

#[test]
fn transform_cache_hit_skips_uncached_upstream() -> Result<()> {
    let source_calls = Arc::new(AtomicUsize::new(0));
    let transform_calls = Arc::new(AtomicUsize::new(0));
    let mut transform = DeterministicTask::builder("test-two-stage-source-v1")
        .function({
            let source_calls = Arc::clone(&source_calls);
            move |input: usize| {
                source_calls.fetch_add(1, Ordering::SeqCst);
                Ok(input * 2)
            }
        })
        .build()?
        .transform("test-two-stage-plus-ten-v1")
        .function({
            let transform_calls = Arc::clone(&transform_calls);
            move |intermediate| {
                transform_calls.fetch_add(1, Ordering::SeqCst);
                Ok(intermediate + 10)
            }
        })
        .cache(ManagedHashCache::<usize>::default())
        .build()?;

    assert_eq!(transform.execute_one(5)?, 20);
    assert_eq!(transform.execute_one(5)?, 20);
    assert_eq!(source_calls.load(Ordering::SeqCst), 1);
    assert_eq!(transform_calls.load(Ordering::SeqCst), 1);

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
                Ok(format!("Result: {}", intermediate))
            })
            .cache(ManagedHashCache::<String>::default())
            .build()?
    };

    let results1 = transform1.with_inputs(0..3).collect::<Result<Vec<_>>>()?;

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
                Ok(format!("Result: {}", intermediate))
            })
            .cache(ManagedHashCache::<String>::default())
            .build()?
    };

    let results2 = transform2.with_inputs(0..3).collect::<Result<Vec<_>>>()?;

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

    let results = transform.with_inputs(0..5u32).collect::<Result<Vec<_>>>()?;

    let expected: Vec<String> = (0..5u32).map(|i| format!("Value: {}", i * 100)).collect();
    assert_eq!(results, expected);

    Ok(())
}

#[test]
fn dependent_input_encodes_param_before_source() {
    let input = DependentInput::new(0x0102u16, 0x0304u16);
    let mut buffer = CanonicalBuffer::new();
    let encoded = buffer.encode(&input);

    assert_eq!(encoded, &[0x01, 0x02, 0x03, 0x04]);
}

#[test]
fn dependent_stochastic_input_encodes_param_source_then_repetition() {
    let input = DependentStochasticInput::new(0x0102u16, 0x0304u16, 5);
    let mut buffer = CanonicalBuffer::new();
    let encoded = buffer.encode(&input);

    assert_eq!(encoded, &[0x01, 0x02, 0x03, 0x04, 0, 0, 0, 0, 0, 0, 0, 5]);
}

fn encode_canonical<T: crate::cache::CanonicalEncode>(value: &T) -> Vec<u8> {
    let mut buffer = CanonicalBuffer::<T>::new();
    buffer.encode(value).to_vec()
}

#[test]
fn dependent_input_inherits_lexicographic_order() {
    let inputs: Vec<DependentInput<i32, i32>> = vec![
        DependentInput::new(i32::MIN, 0),
        DependentInput::new(-1, i32::MAX),
        DependentInput::new(0, i32::MIN),
        DependentInput::new(0, 0),
        DependentInput::new(0, 1),
        DependentInput::new(1, i32::MIN),
        DependentInput::new(i32::MAX, i32::MAX),
    ];
    let encoded: Vec<Vec<u8>> = inputs.iter().map(encode_canonical).collect();
    for (i, window) in encoded.windows(2).enumerate() {
        assert!(
            window[0] < window[1],
            "DependentInput order violated at index {i}: {:?} !< {:?}",
            inputs[i],
            inputs[i + 1]
        );
    }
}

#[test]
fn dependent_stochastic_input_inherits_lexicographic_order() {
    let inputs: Vec<DependentStochasticInput<i16, f64>> = vec![
        DependentStochasticInput::new(i16::MIN, f64::NEG_INFINITY, 0),
        DependentStochasticInput::new(-1, -1.0, u64::MAX),
        DependentStochasticInput::new(0, 0.0, 0),
        DependentStochasticInput::new(0, 0.0, 1),
        DependentStochasticInput::new(0, 1.0, 0),
        DependentStochasticInput::new(1, f64::INFINITY, 0),
        DependentStochasticInput::new(i16::MAX, f64::NAN, u64::MAX),
    ];
    let encoded: Vec<Vec<u8>> = inputs.iter().map(encode_canonical).collect();
    for (i, window) in encoded.windows(2).enumerate() {
        assert!(
            window[0] < window[1],
            "DependentStochasticInput order violated at index {i}: {:?} !< {:?}",
            inputs[i],
            inputs[i + 1]
        );
    }
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
fn parameterized_stochastic_transform_recomputes_only_transform_for_param_changes() -> Result<()> {
    let source_calls = Arc::new(AtomicUsize::new(0));
    let transform_calls = Arc::new(AtomicUsize::new(0));

    let source = DeterministicTask::builder("param-stochastic-source-cache-v1")
        .function({
            let source_calls = Arc::clone(&source_calls);
            move |input: u16| {
                source_calls.fetch_add(1, Ordering::SeqCst);
                Ok(input * 2)
            }
        })
        .cache(ManagedHashCache::<u16>::default())
        .build()?;

    let mut transform = source
        .stochastic_transform("param-stochastic-analysis-v1")
        .function_with_param({
            let transform_calls = Arc::clone(&transform_calls);
            move |rng, value, offset: u16| {
                transform_calls.fetch_add(1, Ordering::SeqCst);
                Ok(value as u64 + offset as u64 + rand::Rng::next_u64(rng))
            }
        })
        .cache(ManagedHashCache::<u64>::default())
        .build()?;

    let input_a = DependentStochasticInput::new(1u16, 10u16, 0);
    let input_b = DependentStochasticInput::new(2u16, 10u16, 0);
    let first_a = transform.execute_one(input_a.clone())?;
    let first_b = transform.execute_one(input_b)?;
    let second_a = transform.execute_one(input_a)?;

    assert_ne!(first_a, first_b);
    assert_eq!(first_a, second_a);
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
    let transform_calls = Arc::new(AtomicUsize::new(0));
    let source = DeterministicTask::builder("source-value-v1")
        .function(|input: u16| Ok(input * 2))
        .cache(ManagedHashCache::<u16>::default())
        .build()?;

    let mut transform = source
        .stochastic_transform("analysis-named-sample-v1")
        .streams(["left", "right"])
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
    let transform_calls = Arc::new(AtomicUsize::new(0));
    let source = DeterministicTask::builder("source-value-v1")
        .function(|input: u16| Ok(input * 2))
        .cache(ManagedHashCache::<u16>::default())
        .build()?;

    let mut transform = source
        .stochastic_transform("analysis-named-resample-v1")
        .streams(["left", "right"])
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
    let transform_calls = Arc::new(AtomicUsize::new(0));
    let source = DeterministicTask::builder("source-value-v1")
        .function(|input: u16| Ok(input * 2))
        .cache(ManagedHashCache::<u16>::default())
        .build()?;

    let mut transform = source
        .stochastic_transform("analysis-named-resample-v1")
        .streams(["left", "right"])
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
