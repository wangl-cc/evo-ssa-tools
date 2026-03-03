use std::marker::PhantomData;

use crate::{CacheStore, CanonicalEncode, Codec, Compute, Result};

pub struct Pipeline<R, F, I, M, O> {
    required: R,
    function: F,
    _phantom: PhantomData<(I, M, O)>,
}

impl<R, F, I, M, O> Pipeline<R, F, I, M, O> {
    pub fn new(required: R, function: F) -> Self {
        Self {
            required,
            function,
            _phantom: PhantomData,
        }
    }
}

impl<R: Clone, F: Clone, I, M, O> Clone for Pipeline<R, F, I, M, O> {
    fn clone(&self) -> Self {
        Self {
            required: self.required.clone(),
            function: self.function.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<R, F, I, M, O, C1, C2> Compute<(C1, C2)> for Pipeline<R, F, I, M, O>
where
    R: Compute<C1, Input = I, Output = M>,
    F: Fn(M) -> Result<O>,
    I: CanonicalEncode,
    M: Codec<Buffer = <O as Codec>::Buffer>,
    O: Codec,
    C1: CacheStore,
    C2: CacheStore,
{
    type Input = I;
    type Output = O;

    fn execute_with_sig(
        &mut self,
        input: Self::Input,
        input_signature: &[u8],
        cache: &(C1, C2),
        codec_buffer: &mut <Self::Output as Codec>::Buffer,
    ) -> Result<Self::Output> {
        let (cache1, cache2) = cache;

        cache2.fetch_or_execute(input_signature, codec_buffer, |codec_buffer| {
            let intermediate =
                self.required
                    .execute_with_sig(input, input_signature, cache1, codec_buffer)?;
            (self.function)(intermediate)
        })
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use std::{
        collections::hash_map::RandomState,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
        thread::sleep,
        time::Duration,
    };

    use rayon::prelude::*;

    use super::*;
    use crate::{CodecBuffer, ExecuteOptions, HashMapStore, SingleStep};

    #[test]
    fn test_pipeline_basic() -> Result<()> {
        // Create a two-stage pipeline: input * 2, then + 10
        let stage1: SingleStep<usize, usize, _> = SingleStep::new(|input| Ok(input * 2));
        let pipeline: Pipeline<_, _, usize, usize, usize> =
            Pipeline::new(stage1, |intermediate| Ok(intermediate + 10));

        let cache1 = HashMapStore::<RandomState>::default();
        let cache2 = HashMapStore::<RandomState>::default();
        let cache = (cache1, cache2);

        let results = pipeline
            .execute_many((0..10).into_par_iter(), &cache, ExecuteOptions::default())?
            .collect::<Result<Vec<usize>>>()?;

        // 0*2+10=10, 1*2+10=12, 2*2+10=14, ...
        let expected: Vec<usize> = (0..10).map(|i| i * 2 + 10).collect();
        assert_eq!(results, expected);

        Ok(())
    }

    #[test]
    fn test_pipeline_two_stage_caching() -> Result<()> {
        let stage1_calls = Arc::new(AtomicUsize::new(0));
        let stage1_calls_clone = stage1_calls.clone();
        let stage2_calls = Arc::new(AtomicUsize::new(0));
        let stage2_calls_clone = stage2_calls.clone();

        let stage1: SingleStep<usize, usize, _> = SingleStep::new(move |input| {
            stage1_calls_clone.fetch_add(1, Ordering::SeqCst);
            sleep(Duration::from_millis(10)); // Simulate work
            Ok(input * 2)
        });

        let pipeline: Pipeline<_, _, usize, usize, usize> =
            Pipeline::new(stage1, move |intermediate| {
                stage2_calls_clone.fetch_add(1, Ordering::SeqCst);
                sleep(Duration::from_millis(10)); // Simulate work
                Ok(intermediate + 10)
            });

        let cache1 = HashMapStore::<RandomState>::default();
        let cache2 = HashMapStore::<RandomState>::default();
        let cache = (cache1, cache2);

        // First execution - both stages should run
        let results1 = pipeline
            .execute_many((0..5).into_par_iter(), &cache, ExecuteOptions::default())?
            .collect::<Result<Vec<usize>>>()?;

        let expected: Vec<usize> = (0..5).map(|i| i * 2 + 10).collect();
        assert_eq!(results1, expected);
        assert_eq!(stage1_calls.load(Ordering::SeqCst), 5);
        assert_eq!(stage2_calls.load(Ordering::SeqCst), 5);

        // Second execution - should use cached final results
        let results2 = pipeline
            .execute_many((0..5).into_par_iter(), &cache, ExecuteOptions::default())?
            .collect::<Result<Vec<usize>>>()?;

        assert_eq!(results2, expected);
        // Neither stage should be called again (both cached)
        assert_eq!(stage1_calls.load(Ordering::SeqCst), 5);
        assert_eq!(stage2_calls.load(Ordering::SeqCst), 5);

        Ok(())
    }

    #[test]
    fn test_pipeline_intermediate_cache_reuse() -> Result<()> {
        let stage1_calls = Arc::new(AtomicUsize::new(0));
        let stage1_calls_clone = stage1_calls.clone();
        let stage2_calls = Arc::new(AtomicUsize::new(0));
        let stage2_calls_clone = stage2_calls.clone();

        let stage1: SingleStep<usize, usize, _> = SingleStep::new(move |input| {
            stage1_calls_clone.fetch_add(1, Ordering::SeqCst);
            sleep(Duration::from_millis(10));
            Ok(input * 2)
        });

        let pipeline: Pipeline<_, _, usize, usize, String> =
            Pipeline::new(stage1, move |intermediate| {
                stage2_calls_clone.fetch_add(1, Ordering::SeqCst);
                sleep(Duration::from_millis(10));
                Ok(format!("Result: {}", intermediate))
            });

        let cache1 = HashMapStore::<RandomState>::default();
        let cache2 = HashMapStore::<RandomState>::default();
        let cache = (cache1, cache2);

        // First execution
        let results1 = pipeline
            .execute_many((0..3).into_par_iter(), &cache, ExecuteOptions::default())?
            .collect::<Result<Vec<String>>>()?;

        let expected: Vec<String> = (0..3).map(|i| format!("Result: {}", i * 2)).collect();
        assert_eq!(results1, expected);
        assert_eq!(stage1_calls.load(Ordering::SeqCst), 3);
        assert_eq!(stage2_calls.load(Ordering::SeqCst), 3);

        // Clear the final results cache but keep intermediate cache
        let cache1 = cache.0; // Keep intermediate cache
        let cache2 = HashMapStore::<RandomState>::default(); // New final cache
        let cache = (cache1, cache2);

        // Second execution - should reuse intermediate results but recompute final
        let results2 = pipeline
            .execute_many((0..3).into_par_iter(), &cache, ExecuteOptions::default())?
            .collect::<Result<Vec<String>>>()?;

        assert_eq!(results2, expected);
        // Stage 1 should not be called again (cached)
        assert_eq!(stage1_calls.load(Ordering::SeqCst), 3);
        // Stage 2 should be called again (cache cleared)
        assert_eq!(stage2_calls.load(Ordering::SeqCst), 6);

        Ok(())
    }

    #[test]
    fn test_pipeline_with_different_types() -> Result<()> {
        // Test pipeline with different input/intermediate/output types
        let stage1: SingleStep<u32, u64, _> = SingleStep::new(|input: u32| Ok(input as u64 * 100));

        let pipeline: Pipeline<_, _, u32, u64, String> =
            Pipeline::new(stage1, |intermediate: u64| {
                Ok(format!("Value: {}", intermediate))
            });

        let cache1 = HashMapStore::<RandomState>::default();
        let cache2 = HashMapStore::<RandomState>::default();
        let cache = (cache1, cache2);

        let results = pipeline
            .execute_many((0..5u32).into_par_iter(), &cache, ExecuteOptions::default())?
            .collect::<Result<Vec<String>>>()?;

        let expected: Vec<String> = (0..5u32).map(|i| format!("Value: {}", i * 100)).collect();
        assert_eq!(results, expected);

        Ok(())
    }

    #[test]
    fn test_pipeline_error_propagation() -> Result<()> {
        // Test that errors in stage1 propagate correctly
        let stage1: SingleStep<usize, usize, _> = SingleStep::new(|input| {
            if input == 3 {
                Err(crate::Error::Interrupted)
            } else {
                Ok(input * 2)
            }
        });

        let mut pipeline: Pipeline<_, _, usize, usize, usize> =
            Pipeline::new(stage1, |intermediate| Ok(intermediate + 10));

        let cache1 = HashMapStore::<RandomState>::default();
        let cache2 = HashMapStore::<RandomState>::default();
        let cache = (cache1, cache2);
        let mut encode_buffer = vec![0u8; usize::SIZE];
        let mut codec_buffer = <usize as Codec>::Buffer::init();

        // Should succeed for non-3 values
        let result =
            unsafe { pipeline.execute(2, &cache, &mut encode_buffer, &mut codec_buffer)? };
        assert_eq!(result, 14); // 2*2+10

        // Should fail for 3
        let result = unsafe { pipeline.execute(3, &cache, &mut encode_buffer, &mut codec_buffer) };
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_pipeline_stage2_error_propagation() -> Result<()> {
        // Test that errors in stage2 propagate correctly
        let stage1: SingleStep<usize, usize, _> = SingleStep::new(|input| Ok(input * 2));

        let mut pipeline: Pipeline<_, _, usize, usize, usize> =
            Pipeline::new(stage1, |intermediate| {
                if intermediate == 6 {
                    // 3 * 2 = 6
                    Err(crate::Error::Interrupted)
                } else {
                    Ok(intermediate + 10)
                }
            });

        let cache1 = HashMapStore::<RandomState>::default();
        let cache2 = HashMapStore::<RandomState>::default();
        let cache = (cache1, cache2);
        let mut encode_buffer = vec![0u8; usize::SIZE];
        let mut codec_buffer = <usize as Codec>::Buffer::init();

        // Should succeed for values != 3
        let result =
            unsafe { pipeline.execute(2, &cache, &mut encode_buffer, &mut codec_buffer)? };
        assert_eq!(result, 14); // 2*2+10

        // Should fail for 3 (produces 6 in stage1)
        let result = unsafe { pipeline.execute(3, &cache, &mut encode_buffer, &mut codec_buffer) };
        assert!(result.is_err());

        Ok(())
    }
}
