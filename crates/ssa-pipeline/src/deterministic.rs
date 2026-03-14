use std::marker::PhantomData;

use crate::{
    CacheStore, CanonicalEncode, Compute, Result,
    cache::{
        codec::{CodecEngine, EngineFactory},
        storage::WorkerForkStore,
    },
};

/// Deterministic compute node.
///
/// `DeterministicStep` is a memoized `input -> output` mapping where the output is a pure
/// function of the input (no randomness, no external state).
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
/// # use ssa_pipeline::prelude::*;
/// # use rayon::prelude::*;
/// # fn main() -> ssa_pipeline::error::Result<()> {
/// type Store = HashMapStore<std::collections::hash_map::RandomState>;
/// let step = DeterministicStep::new(Store::default(), |i: i32| Ok(i.abs()), Bitcode06::default);
/// let results = step
///     .execute_many((0..10).into_par_iter(), ExecuteOptions::default())?
///     .collect::<ssa_pipeline::error::Result<Vec<i32>>>()?;
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
/// output type in an incompatible way, use a fresh keyspace (e.g. a different store wrapper).
///
/// To disable caching entirely, pass `()` as the cache.
#[derive(Debug)]
pub struct DeterministicStep<C, I, O, F, EF> {
    cache: C,
    function: F,
    engine_factory: EF,
    _phantom: PhantomData<(I, O)>,
}

impl<C, I, O, F, EF> DeterministicStep<C, I, O, F, EF>
where
    F: Fn(I) -> Result<O>,
    EF: EngineFactory,
    EF::Engine: CodecEngine<O>,
{
    /// Create a deterministic step from `cache` and `function`.
    ///
    /// Pass `()` as `cache` to disable caching.
    ///
    /// `cache` should be dedicated to this step (see cache keyspace contract).
    pub fn new(cache: C, function: F, engine_factory: EF) -> Self {
        Self {
            cache,
            function,
            engine_factory,
            _phantom: PhantomData,
        }
    }
}

impl<C: WorkerForkStore, I, O, F: Clone, EF: Clone> Clone for DeterministicStep<C, I, O, F, EF> {
    fn clone(&self) -> Self {
        Self {
            cache: self.cache.fork_store(),
            function: self.function.clone(),
            engine_factory: self.engine_factory.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<C, I, O, F, EF> Compute for DeterministicStep<C, I, O, F, EF>
where
    F: Fn(I) -> Result<O>,
    C: CacheStore,
    I: CanonicalEncode,
    EF: EngineFactory,
    EF::Engine: CodecEngine<O>,
{
    type Engine = EF::Engine;
    type Input = I;
    type Output = O;

    fn make_engine(&self) -> Self::Engine {
        self.engine_factory.make_engine()
    }

    fn execute_with_encoded_input(
        &mut self,
        input: Self::Input,
        encoded: &[u8],
        engine: &mut Self::Engine,
    ) -> Result<Self::Output> {
        let cache = &self.cache;
        let function = &self.function;
        cache.fetch_or_execute(encoded, engine, |_| function(input))
    }
}

#[cfg(test)]
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
        cache::codec::{Error as CodecError, SkipReason, fixtures::FixtureEngine},
        prelude::*,
        test_utils::execute_one,
    };

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

    impl CodecEngine<usize> for TaggedUsizeEngine {
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

    #[test]
    fn test_deterministic_caching() -> Result<()> {
        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count_clone = call_count.clone();

        let compute = DeterministicStep::new(
            DefaultHashMapStore::default(),
            move |i: usize| {
                call_count_clone.fetch_add(1, Ordering::SeqCst);
                Ok(i * 3)
            },
            FixtureEngine::default,
        );

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
        let compute = DeterministicStep::new(
            DefaultHashMapStore::default(),
            |i: usize| {
                sleep(Duration::from_millis(100));
                Ok(i + 1)
            },
            FixtureEngine::default,
        );

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
        let mut compute = DeterministicStep::new(
            DefaultHashMapStore::default(),
            |i: usize| {
                if i == 5 {
                    Err(crate::error::Error::Interrupted)
                } else {
                    Ok(i * 2)
                }
            },
            FixtureEngine::default,
        );

        let result = execute_one(&mut compute, 3)?;
        assert_eq!(result, 6);

        let result = execute_one(&mut compute, 5);
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_deterministic_execution_order() -> Result<()> {
        let compute = DeterministicStep::new(
            DefaultHashMapStore::default(),
            |i: usize| {
                sleep(Duration::from_millis(20 - (i % 20) as u64));
                Ok(i + 100)
            },
            FixtureEngine::default,
        );

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

        let mut compute = DeterministicStep::new(
            (),
            move |i: usize| {
                call_count_clone.fetch_add(1, Ordering::SeqCst);
                Ok(i * 11)
            },
            FixtureEngine::default,
        );

        let output1 = execute_one(&mut compute, 3)?;
        let output2 = execute_one(&mut compute, 3)?;

        assert_eq!(output1, 33);
        assert_eq!(output2, 33);
        assert_eq!(call_count.load(Ordering::SeqCst), 2);
        Ok(())
    }

    #[test]
    fn test_explicit_shared_cache_behavior() -> Result<()> {
        let call_count = Arc::new(AtomicUsize::new(0));

        let mut compute_a = DeterministicStep::new(
            DefaultHashMapStore::default(),
            {
                let call_count = call_count.clone();
                move |i: usize| {
                    call_count.fetch_add(1, Ordering::SeqCst);
                    Ok(i * 7)
                }
            },
            FixtureEngine::default,
        );
        let mut compute_b = compute_a.clone();

        let output_a = execute_one(&mut compute_a, 4usize)?;
        let output_b = execute_one(&mut compute_b, 4usize)?;

        assert_eq!(output_a, 28);
        assert_eq!(output_b, 28);
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[test]
    fn test_engine_factory_supports_non_default_non_clone_engine() -> Result<()> {
        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count_clone = call_count.clone();

        let compute = DeterministicStep::new(
            DefaultHashMapStore::default(),
            move |i: usize| {
                call_count_clone.fetch_add(1, Ordering::SeqCst);
                Ok(i + 10)
            },
            || TaggedUsizeEngine::new(0xA5),
        );

        let outputs1 = compute
            .execute_many((0..8usize).into_par_iter(), ExecuteOptions::default())?
            .collect::<Result<Vec<usize>>>()?;
        let outputs2 = compute
            .execute_many((0..8usize).into_par_iter(), ExecuteOptions::default())?
            .collect::<Result<Vec<usize>>>()?;

        let expected: Vec<_> = (0..8usize).map(|i| i + 10).collect();
        assert_eq!(outputs1, expected);
        assert_eq!(outputs2, expected);
        assert_eq!(call_count.load(Ordering::SeqCst), 8);
        Ok(())
    }

    #[test]
    fn test_engine_factory_can_capture_runtime_configuration() -> Result<()> {
        let tag = 0x3C;
        let mut compute = DeterministicStep::new(
            DefaultHashMapStore::default(),
            |i: usize| Ok(i * 5),
            move || TaggedUsizeEngine::new(tag),
        );

        assert_eq!(execute_one(&mut compute, 4usize)?, 20);
        assert_eq!(execute_one(&mut compute, 4usize)?, 20);
        Ok(())
    }
}
