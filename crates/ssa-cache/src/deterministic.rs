use std::marker::PhantomData;

use crate::{CacheStore, CanonicalEncode, Compute, Result, cache::codec::CodecEngine};

/// Deterministic compute node.
///
/// `DeterministicStep` is a cached `input -> output` mapping where the output is a pure function
/// of the input (no randomness, no external state).
///
/// # When to use
///
/// - The output depends only on the input value.
/// - You want to reuse results across repeated calls / parameter sweeps.
/// - You want parallel batch execution via [`Compute::execute_many`].
///
/// # Example
///
/// ```rust
/// # use ssa_cache::prelude::*;
/// # use rayon::prelude::*;
/// # fn main() -> ssa_cache::error::Result<()> {
/// type Store = HashMapStore<std::collections::hash_map::RandomState>;
/// let step = DeterministicStep::new(Store::default(), |i: i32| Ok(i.abs()));
/// let results = step
///     .execute_many((0..10).into_par_iter(), ExecuteOptions::default())?
///     .collect::<ssa_cache::error::Result<Vec<i32>>>()?;
/// # Ok(())
/// # }
/// ```
///
/// # Caching / keyspace contract
///
/// The `cache` passed to [`DeterministicStep::new`] is treated as this step's private keyspace.
/// Reusing the same underlying keyspace for different compute logic is not supported and may
/// return values written by another computation for identical input bytes.
///
/// Keyspace compatibility is caller-managed: if you change the compute logic, input encoding, or
/// output type in an incompatible way, use a fresh keyspace (e.g. a different store, or a
/// different `fjall::Keyspace`).
///
/// To disable caching entirely, pass `()` as the cache.
#[derive(Debug)]
pub struct DeterministicStep<C, I, O, E, F> {
    cache: C,
    function: F,
    _phantom: PhantomData<(I, O, E)>,
}

impl<C, I, O, E, F> DeterministicStep<C, I, O, E, F>
where
    F: Fn(I) -> Result<O>,
{
    /// Create a deterministic step from `cache` and `function` with explicit engine type `E`.
    pub fn new_with_engine(cache: C, function: F) -> Self {
        Self {
            cache,
            function,
            _phantom: PhantomData,
        }
    }
}

#[cfg(feature = "bitcode")]
impl<C, I, O, F> DeterministicStep<C, I, O, crate::cache::codec::bitcode::Bitcode, F>
where
    F: Fn(I) -> Result<O>,
{
    /// Create a deterministic step from `cache` and `function` using bitcode engine.
    pub fn new(cache: C, function: F) -> Self {
        Self::new_with_engine(cache, function)
    }
}

impl<C: Clone, I, O, E, F: Clone> Clone for DeterministicStep<C, I, O, E, F> {
    fn clone(&self) -> Self {
        Self {
            cache: self.cache.clone(),
            function: self.function.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<C, I, O, E, F> Compute for DeterministicStep<C, I, O, E, F>
where
    F: Fn(I) -> Result<O>,
    C: CacheStore,
    I: CanonicalEncode,
    E: CodecEngine<O>,
{
    type Engine = E;
    type Input = I;
    type Output = O;

    fn execute_with_encoded_input(
        &mut self,
        input: Self::Input,
        encoded: &[u8],
        engine: &mut Self::Engine,
    ) -> Result<Self::Output> {
        let cache = &self.cache;
        let function = &self.function;
        cache.fetch_or_execute::<O, E, _>(encoded, engine, |_| function(input))
    }
}

#[cfg(test)]
#[cfg(feature = "bitcode")]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use std::{
        sync::{
            Arc,
            atomic::{AtomicBool, AtomicUsize, Ordering},
        },
        thread::{sleep, spawn},
        time::Duration,
    };

    use rayon::prelude::*;

    use super::*;
    use crate::{
        cache::{codec::bitcode::Bitcode, storage::DefaultHashMapStore},
        prelude::*,
    };

    #[test]
    fn test_deterministic_basic_with_owned_cache() -> Result<()> {
        let mut compute: DeterministicStep<_, usize, usize, Bitcode, _> =
            DeterministicStep::new(DefaultHashMapStore::default(), |i| Ok(i + 1));
        let mut encode_buffer = vec![0u8; usize::SIZE];
        let mut engine = Bitcode::default();

        let result = unsafe { compute.execute(5, &mut encode_buffer, &mut engine)? };
        assert_eq!(result, 6);

        Ok(())
    }

    #[test]
    fn test_execute_many_without_runtime_cache_arg() -> Result<()> {
        let compute: DeterministicStep<_, usize, usize, Bitcode, _> =
            DeterministicStep::new(DefaultHashMapStore::default(), |i| Ok(i * 2));

        let results = compute
            .execute_many((0..100).into_par_iter(), ExecuteOptions::default())?
            .collect::<Result<Vec<usize>>>()?;

        assert_eq!(results, (0..100).map(|i| i * 2).collect::<Vec<usize>>());
        Ok(())
    }

    #[test]
    fn test_deterministic_caching() -> Result<()> {
        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count_clone = call_count.clone();

        let compute: DeterministicStep<_, usize, usize, Bitcode, _> =
            DeterministicStep::new(DefaultHashMapStore::default(), move |i| {
                call_count_clone.fetch_add(1, Ordering::SeqCst);
                Ok(i * 3)
            });

        let n_inputs = 5;

        let results1 = compute
            .execute_many((0..n_inputs).into_par_iter(), ExecuteOptions::default())?
            .collect::<Result<Vec<usize>>>()?;

        let expected: Vec<usize> = (0..n_inputs).map(|i| i * 3).collect();
        assert_eq!(results1, expected);
        assert_eq!(call_count.load(Ordering::SeqCst), n_inputs);

        let results2 = compute
            .execute_many((0..n_inputs).into_par_iter(), ExecuteOptions::default())?
            .collect::<Result<Vec<usize>>>()?;

        assert_eq!(results2, expected);
        assert_eq!(call_count.load(Ordering::SeqCst), n_inputs);

        Ok(())
    }

    #[test]
    fn test_deterministic_interrupt() -> Result<()> {
        let compute: DeterministicStep<_, usize, usize, Bitcode, _> =
            DeterministicStep::new(DefaultHashMapStore::default(), |i| {
                sleep(Duration::from_millis(100));
                Ok(i + 1)
            });

        let interrupted = Arc::new(AtomicBool::new(false));
        let interrupted_clone = interrupted.clone();

        let handle = spawn(move || {
            compute
                .execute_many(
                    (0..100).into_par_iter(),
                    ExecuteOptions::with_interrupt_signal(interrupted),
                )?
                .collect::<Result<Vec<usize>>>()
        });

        sleep(Duration::from_millis(50));
        interrupted_clone.store(true, Ordering::Release);

        let results = handle.join().expect("Failed to join thread");
        assert!(matches!(
            results.unwrap_err(),
            crate::error::Error::Interrupted
        ));

        Ok(())
    }

    #[test]
    fn test_deterministic_error_propagation() -> Result<()> {
        let mut compute: DeterministicStep<_, usize, usize, Bitcode, _> =
            DeterministicStep::new(DefaultHashMapStore::default(), |i| {
                if i == 5 {
                    Err(crate::error::Error::Interrupted)
                } else {
                    Ok(i * 2)
                }
            });
        let mut encode_buffer = vec![0u8; usize::SIZE];
        let mut engine = Bitcode::default();

        let result = unsafe { compute.execute(3, &mut encode_buffer, &mut engine)? };
        assert_eq!(result, 6);

        let result = unsafe { compute.execute(5, &mut encode_buffer, &mut engine) };
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_deterministic_execution_order() -> Result<()> {
        let compute: DeterministicStep<_, usize, usize, Bitcode, _> =
            DeterministicStep::new(DefaultHashMapStore::default(), |i| {
                sleep(Duration::from_millis(20 - (i % 20) as u64));
                Ok(i + 100)
            });

        let n_inputs = 20;

        let results = compute
            .execute_many((0..n_inputs).into_par_iter(), ExecuteOptions::default())?
            .collect::<Result<Vec<usize>>>()?;

        assert_eq!(
            results,
            (0..n_inputs).map(|i| i + 100).collect::<Vec<usize>>()
        );

        Ok(())
    }

    #[test]
    fn test_no_cache_mode_still_works() -> Result<()> {
        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count_clone = call_count.clone();

        let mut compute: DeterministicStep<_, usize, usize, Bitcode, _> =
            DeterministicStep::new((), move |i| {
                call_count_clone.fetch_add(1, Ordering::SeqCst);
                Ok(i * 11)
            });
        let mut encode_buffer = vec![0u8; usize::SIZE];
        let mut engine = Bitcode::default();

        let output1 = unsafe { compute.execute(3, &mut encode_buffer, &mut engine)? };
        let output2 = unsafe { compute.execute(3, &mut encode_buffer, &mut engine)? };

        assert_eq!(output1, 33);
        assert_eq!(output2, 33);
        assert_eq!(call_count.load(Ordering::SeqCst), 2);
        Ok(())
    }

    #[test]
    fn test_explicit_shared_cache_behavior() -> Result<()> {
        let shared_cache = DefaultHashMapStore::default();
        let call_count = Arc::new(AtomicUsize::new(0));

        let mut compute_a = {
            let call_count = call_count.clone();
            DeterministicStep::new(shared_cache.clone(), move |i| {
                call_count.fetch_add(1, Ordering::SeqCst);
                Ok(i * 7)
            })
        };

        let mut compute_b = {
            let call_count = call_count.clone();
            DeterministicStep::new(shared_cache, move |i| {
                call_count.fetch_add(1, Ordering::SeqCst);
                Ok(i * 7)
            })
        };

        let mut encode_buffer_a = vec![0u8; usize::SIZE];
        let mut engine_a = Bitcode::default();
        let mut encode_buffer_b = vec![0u8; usize::SIZE];
        let mut engine_b = Bitcode::default();

        let output_a = unsafe { compute_a.execute(4usize, &mut encode_buffer_a, &mut engine_a)? };
        let output_b = unsafe { compute_b.execute(4usize, &mut encode_buffer_b, &mut engine_b)? };

        assert_eq!(output_a, 28);
        assert_eq!(output_b, 28);
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
        Ok(())
    }
}
