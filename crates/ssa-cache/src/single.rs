use std::marker::PhantomData;

use crate::{CacheStore, CanonicalEncode, Codec, Compute, Result};

#[derive(Debug)]
pub struct SingleStep<I, O, F> {
    function: F,
    _phantom: PhantomData<(I, O)>,
}

impl<I, O, F> SingleStep<I, O, F>
where
    F: Fn(I) -> Result<O>,
{
    /// Create a new pure (deterministic) cached computation
    ///
    /// The function can return a `Result` to indicate computation failure.
    pub fn new(function: F) -> Self {
        Self {
            function,
            _phantom: PhantomData,
        }
    }
}

impl<I, O, F: Clone> Clone for SingleStep<I, O, F> {
    fn clone(&self) -> Self {
        Self {
            function: self.function.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<I, O, F, C> Compute<C> for SingleStep<I, O, F>
where
    F: Fn(I) -> Result<O>,
    C: CacheStore,
    I: CanonicalEncode,
    O: Codec,
{
    type Input = I;
    type Output = O;

    fn execute_with_sig(
        &mut self,
        input: Self::Input,
        input_signature: &[u8],
        cache: &C,
        codec_buffer: &mut <Self::Output as Codec>::Buffer,
    ) -> Result<Self::Output> {
        cache.fetch_or_execute(input_signature, codec_buffer, |_| (self.function)(input))
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use std::{
        collections::hash_map::RandomState,
        sync::{
            Arc,
            atomic::{AtomicBool, AtomicUsize, Ordering},
        },
        thread::{sleep, spawn},
        time::Duration,
    };

    use rayon::prelude::*;

    use super::*;
    use crate::{CodecBuffer, ExecuteOptions, HashMapStore};

    #[test]
    fn test_single_step_basic() -> Result<()> {
        let mut compute: SingleStep<usize, usize, _> = SingleStep::new(|i| Ok(i + 1));
        let cache = HashMapStore::<RandomState>::default();
        let mut encode_buffer = vec![0u8; usize::SIZE];
        let mut codec_buffer = <usize as Codec>::Buffer::init();

        // Execute and check result
        let result = unsafe { compute.execute(5, &cache, &mut encode_buffer, &mut codec_buffer)? };
        assert_eq!(result, 6);

        Ok(())
    }

    #[test]
    fn test_single_step_parallel() -> Result<()> {
        let compute: SingleStep<usize, usize, _> = SingleStep::new(|i| Ok(i * 2));
        let cache = HashMapStore::<RandomState>::default();

        let results = compute
            .execute_many((0..100).into_par_iter(), &cache, ExecuteOptions::default())?
            .collect::<Result<Vec<usize>>>()?;

        assert_eq!(results, (0..100).map(|i| i * 2).collect::<Vec<usize>>());

        Ok(())
    }

    #[test]
    fn test_single_step_caching() -> Result<()> {
        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count_clone = call_count.clone();

        let compute: SingleStep<usize, usize, _> = SingleStep::new(move |i| {
            call_count_clone.fetch_add(1, Ordering::SeqCst);
            Ok(i * 3)
        });

        let cache = HashMapStore::<RandomState>::default();
        let n_inputs = 5;

        // First execution - should call function for each input
        let results1 = compute
            .execute_many(
                (0..n_inputs).into_par_iter(),
                &cache,
                ExecuteOptions::default(),
            )?
            .collect::<Result<Vec<usize>>>()?;

        let expected: Vec<usize> = (0..n_inputs).map(|i| i * 3).collect();
        assert_eq!(results1, expected);
        assert_eq!(call_count.load(Ordering::SeqCst), n_inputs);

        // Second execution - should use cached results (no additional calls)
        let results2 = compute
            .execute_many(
                (0..n_inputs).into_par_iter(),
                &cache,
                ExecuteOptions::default(),
            )?
            .collect::<Result<Vec<usize>>>()?;

        assert_eq!(results2, expected);
        // Should still be the same number of calls (cached)
        assert_eq!(call_count.load(Ordering::SeqCst), n_inputs);

        Ok(())
    }

    #[test]
    fn test_single_step_interrupt() -> Result<()> {
        let compute: SingleStep<usize, usize, _> = SingleStep::new(|i| {
            sleep(Duration::from_millis(100));
            Ok(i + 1)
        });
        let cache = HashMapStore::<RandomState>::default();

        let interrupted = Arc::new(AtomicBool::new(false));
        let interrupted_clone = interrupted.clone();

        let handle = spawn(move || {
            compute
                .execute_many(
                    (0..100).into_par_iter(),
                    &cache,
                    ExecuteOptions::with_interrupt_signal(interrupted),
                )?
                .collect::<Result<Vec<usize>>>()
        });

        sleep(Duration::from_millis(50));
        interrupted_clone.store(true, Ordering::Relaxed);

        let results = handle.join().expect("Failed to join thread");
        assert!(matches!(results.unwrap_err(), crate::Error::Interrupted));

        Ok(())
    }

    #[test]
    fn test_single_step_error_propagation() -> Result<()> {
        let mut compute: SingleStep<usize, usize, _> = SingleStep::new(|i| {
            if i == 5 {
                Err(crate::Error::Interrupted) // Use any error type
            } else {
                Ok(i * 2)
            }
        });
        let cache = HashMapStore::<RandomState>::default();
        let mut encode_buffer = vec![0u8; usize::SIZE];
        let mut codec_buffer = <usize as Codec>::Buffer::init();

        // Should succeed for non-5 values
        let result = unsafe { compute.execute(3, &cache, &mut encode_buffer, &mut codec_buffer)? };
        assert_eq!(result, 6);

        // Should fail for 5
        let result = unsafe { compute.execute(5, &cache, &mut encode_buffer, &mut codec_buffer) };
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_single_step_execution_order() -> Result<()> {
        // Test that parallel execution maintains input order in results
        // Use varying sleep times based on input to ensure out-of-order completion
        let compute: SingleStep<usize, usize, _> = SingleStep::new(|i| {
            // Longer sleep for lower numbers to ensure they finish later
            sleep(Duration::from_millis(20 - (i % 20) as u64));
            Ok(i + 100)
        });

        let cache = HashMapStore::<RandomState>::default();
        let n_inputs = 20;

        let results = compute
            .execute_many(
                (0..n_inputs).into_par_iter(),
                &cache,
                ExecuteOptions::default(),
            )?
            .collect::<Result<Vec<usize>>>()?;

        // Results should be in input order despite different completion times
        assert_eq!(
            results,
            (0..n_inputs).map(|i| i + 100).collect::<Vec<usize>>()
        );

        Ok(())
    }
}
