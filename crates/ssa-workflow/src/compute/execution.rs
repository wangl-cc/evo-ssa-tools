//! Core execution traits and batch execution builders.

use std::sync::{Arc, atomic};

use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use crate::{
    cache::CanonicalEncode,
    error::{Error, Result},
    identity::ComputationPath,
};

/// Core trait for execution nodes.
///
/// A `Compute` node maps an input to an output, optionally using a cache.
///
/// # Cache key
///
/// `Input` is encoded into canonical bytes via [`CanonicalEncode`]. A schema signature prefix plus
/// those payload bytes form the cache key.
/// If you change the meaning of your input encoding or the semantics of the computation, treat it
/// as a new keyspace.
pub trait Compute {
    /// Input type that can be canonical-encoded into a cache key.
    type Input: CanonicalEncode;
    /// Output type emitted by this node.
    type Output;

    /// Return the computation path used for cache namespace and downstream transforms.
    fn computation_path(&self) -> &ComputationPath;

    /// Low-level API for single execution with pre-encoded input bytes.
    ///
    /// For implementers: `encoded` is the canonical encoding of `input` and is the cache key.
    /// Implementations should avoid re-encoding `input` and use `encoded` directly.
    fn execute_with_encoded_input(
        &mut self,
        input: Self::Input,
        encoded: &[u8],
    ) -> Result<Self::Output>;

    /// Low-level API for single execution with a pre-allocated encode buffer.
    ///
    /// Most users should prefer [`Self::execute_one`].
    ///
    /// # Safety
    ///
    /// The buffer must have length at least `Self::Input::KEY_SIZE`.
    /// Implementations should only access `buffer[..Self::Input::KEY_SIZE]`.
    ///
    /// See [`CanonicalEncode`] for more details.
    unsafe fn execute_one_with_buffer(
        &mut self,
        input: Self::Input,
        encode_buffer: &mut [u8],
    ) -> Result<Self::Output> {
        // Safety: The safety is guaranteed by the caller.
        let encoded = unsafe { input.encode_key_with_buffer(encode_buffer) };
        self.execute_with_encoded_input(input, encoded)
    }

    /// Execute computation for one input.
    ///
    /// This is the safe, single-item counterpart to [`Self::execute_one_with_buffer`].
    ///
    /// If you need to execute multiple ordered inputs in parallel, use [`Self::with_inputs`]
    /// instead.
    fn execute_one(&mut self, input: Self::Input) -> Result<Self::Output> {
        let mut encode_buffer = vec![0u8; Self::Input::KEY_SIZE];
        // Safety: The buffer is initialized with length Self::Input::KEY_SIZE.
        unsafe { self.execute_one_with_buffer(input, &mut encode_buffer) }
    }

    /// Start a batch execution builder for ordered `inputs`.
    ///
    /// Inputs must produce an [`IndexedParallelIterator`], so collected outputs follow the logical
    /// input order and Rayon can use its indexed collection path.
    ///
    /// Unordered parallel inputs such as `HashSet` intentionally do not satisfy this API:
    ///
    /// ```compile_fail
    /// # use ssa_workflow::prelude::*;
    /// # use std::collections::HashSet;
    /// # fn main() -> ssa_workflow::Result<()> {
    /// let task = DeterministicTask::builder("unordered-inputs-v1")
    ///     .function(|input: u8| Ok(input))
    ///     .build()?;
    /// let inputs = HashSet::from([1u8, 2u8]);
    /// let _ = task.with_inputs(inputs);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Each input item is scheduled independently. Batch execution does not deduplicate equal
    /// inputs or provide single-flight behavior for equal canonical keys; if duplicate inputs
    /// are executed concurrently, they may compute the same cache miss more than once before
    /// one worker stores the result.
    fn with_inputs<I>(&self, inputs: I) -> BatchExecution<'_, Self, I>
    where
        Self: Sized,
        I: IntoParallelIterator<Item = Self::Input>,
        I::Iter: IndexedParallelIterator<Item = Self::Input>,
    {
        BatchExecution {
            compute: self,
            inputs,
            interrupt_signal: None,
        }
    }
}

/// Batch execution builder returned by [`Compute::with_inputs`].
pub struct BatchExecution<'a, C, I>
where
    C: Compute,
    I: IntoParallelIterator<Item = C::Input>,
    I::Iter: IndexedParallelIterator<Item = C::Input>,
{
    compute: &'a C,
    inputs: I,
    interrupt_signal: Option<InterruptSignal>,
}

impl<'a, C, I> BatchExecution<'a, C, I>
where
    C: Compute,
    I: IntoParallelIterator<Item = C::Input>,
    I::Iter: IndexedParallelIterator<Item = C::Input>,
{
    /// Interrupt pending work when `signal` is set.
    ///
    /// Items already executing are not forcibly cancelled.
    pub fn with_interrupt_signal(mut self, signal: InterruptSignal) -> Self {
        self.interrupt_signal = Some(signal);
        self
    }

    /// Build the indexed parallel iterator for this batch.
    ///
    /// Collecting this iterator as `Result<Vec<_>>` naturally short-circuits on errors.
    /// Collecting it as `Vec<Result<_>>` keeps successful and failed items.
    pub fn execute(self) -> impl IndexedParallelIterator<Item = Result<C::Output>> + 'a
    where
        C: Clone + Sync + 'a,
        I: 'a,
        C::Output: Send,
    {
        let signal = self.interrupt_signal;
        let compute = self.compute;
        self.inputs.into_par_iter().map_init(
            move || {
                (
                    vec![0u8; C::Input::KEY_SIZE],
                    compute.clone(),
                    signal.clone(),
                )
            },
            |(buffer, compute, signal), input| {
                if let Some(signal) = signal
                    && signal.is_interrupted()
                {
                    return Err(Error::Interrupted);
                }

                // Safety: The buffer is initialized with length Self::Input::KEY_SIZE.
                unsafe { compute.execute_one_with_buffer(input, buffer) }
            },
        )
    }

    /// Execute the batch and collect outputs in input order.
    ///
    /// This is the natural fail-fast collection mode: collection returns the first observed error.
    pub fn collect(self) -> Result<Vec<C::Output>>
    where
        C: Clone + Sync + 'a,
        I: 'a,
        C::Output: Send,
    {
        self.execute().collect()
    }

    /// Execute the batch and collect every item result.
    ///
    /// Use this for long sweeps where independent failures should not stop collection.
    pub fn results(self) -> Vec<Result<C::Output>>
    where
        C: Clone + Sync + 'a,
        I: 'a,
        C::Output: Send,
    {
        self.execute().collect()
    }
}

/// Shared interrupt signal for batch execution.
#[derive(Debug, Clone, Default)]
pub struct InterruptSignal {
    interrupted: Arc<atomic::AtomicBool>,
}

impl InterruptSignal {
    /// Create a non-interrupted signal.
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark this signal as interrupted.
    pub fn interrupt(&self) {
        self.interrupted.store(true, atomic::Ordering::Release);
    }

    /// Clear the interrupt flag.
    pub fn reset(&self) {
        self.interrupted.store(false, atomic::Ordering::Release);
    }

    /// Return whether this signal has been interrupted.
    pub fn is_interrupted(&self) -> bool {
        self.interrupted.load(atomic::Ordering::Acquire)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use std::sync::{
        Arc, LazyLock,
        atomic::{AtomicUsize, Ordering},
    };

    use super::*;

    static TEST_PATH: LazyLock<ComputationPath> =
        LazyLock::new(|| ComputationPath::root_from_str("test-compute-v1"));

    mod single_item {
        use super::*;

        #[derive(Clone)]
        struct EncodedEcho {
            calls: Arc<AtomicUsize>,
        }

        impl Compute for EncodedEcho {
            type Input = u16;
            type Output = Vec<u8>;

            fn computation_path(&self) -> &ComputationPath {
                &TEST_PATH
            }

            fn execute_with_encoded_input(
                &mut self,
                _input: Self::Input,
                encoded: &[u8],
            ) -> Result<Self::Output> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                Ok(encoded.to_vec())
            }
        }

        #[test]
        fn execute_one_with_buffer_uses_canonical_input_bytes() -> Result<()> {
            let calls = Arc::new(AtomicUsize::new(0));
            let mut compute = EncodedEcho {
                calls: Arc::clone(&calls),
            };
            let mut buffer = vec![0xAA; u16::KEY_SIZE + 1];

            // Safety: the buffer is longer than the required u16 canonical key encoding.
            let output = unsafe { compute.execute_one_with_buffer(0x1234, &mut buffer) }?;

            let mut expected = Vec::new();
            expected.extend_from_slice(&u16::SCHEMA_SIGNATURE.to_be_bytes());
            expected.extend_from_slice(&0x1234u16.to_be_bytes());
            assert_eq!(output, expected);
            assert_eq!(&buffer[..u16::KEY_SIZE], expected);
            assert_eq!(buffer[u16::KEY_SIZE], 0xAA);
            assert_eq!(calls.load(Ordering::SeqCst), 1);
            Ok(())
        }
    }

    mod batch_results {
        use super::*;

        #[derive(Debug, thiserror::Error)]
        #[error("test compute failed")]
        struct TestComputeError;

        #[derive(Clone)]
        struct FallibleCompute;

        impl Compute for FallibleCompute {
            type Input = u8;
            type Output = u8;

            fn computation_path(&self) -> &ComputationPath {
                &TEST_PATH
            }

            fn execute_with_encoded_input(
                &mut self,
                input: Self::Input,
                _encoded: &[u8],
            ) -> Result<Self::Output> {
                if input == 2 {
                    Err(Error::Compute(Box::new(TestComputeError)))
                } else {
                    Ok(input + 10)
                }
            }
        }

        #[test]
        fn results_collects_every_item_result_without_outer_result() {
            let results = FallibleCompute.with_inputs(0u8..4).results();

            assert_eq!(results.len(), 4);
            assert_eq!(results[0].as_ref().ok(), Some(&10));
            assert_eq!(results[1].as_ref().ok(), Some(&11));
            assert!(matches!(results[2], Err(Error::Compute(_))));
            assert_eq!(results[3].as_ref().ok(), Some(&13));
        }
    }

    mod interrupt_signal {
        use super::*;

        #[test]
        fn signal_is_shared_and_resettable() {
            let signal = InterruptSignal::new();
            let cloned = signal.clone();

            assert!(!signal.is_interrupted());
            cloned.interrupt();
            assert!(signal.is_interrupted());
            signal.reset();
            assert!(!cloned.is_interrupted());
        }
    }

    mod interrupt_scheduling {
        use super::*;

        #[derive(Clone)]
        struct CountingCompute {
            calls: Arc<AtomicUsize>,
        }

        impl Compute for CountingCompute {
            type Input = u8;
            type Output = u8;

            fn computation_path(&self) -> &ComputationPath {
                &TEST_PATH
            }

            fn execute_with_encoded_input(
                &mut self,
                input: Self::Input,
                _encoded: &[u8],
            ) -> Result<Self::Output> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                Ok(input)
            }
        }

        #[test]
        fn signal_set_before_batch_skips_all_work() {
            let signal = InterruptSignal::new();
            signal.interrupt();
            let calls = Arc::new(AtomicUsize::new(0));

            let results = CountingCompute {
                calls: Arc::clone(&calls),
            }
            .with_inputs(0u8..4)
            .with_interrupt_signal(signal)
            .results();

            assert_eq!(calls.load(Ordering::SeqCst), 0);
            assert_eq!(results.len(), 4);
            assert!(
                results
                    .iter()
                    .all(|result| { matches!(result, Err(Error::Interrupted)) })
            );
        }

        #[test]
        fn collect_observes_preset_interrupt_signal() {
            let signal = InterruptSignal::new();
            signal.interrupt();
            let calls = Arc::new(AtomicUsize::new(0));

            let result = CountingCompute {
                calls: Arc::clone(&calls),
            }
            .with_inputs(0u8..4)
            .with_interrupt_signal(signal)
            .collect();

            assert!(matches!(result, Err(Error::Interrupted)));
            assert_eq!(calls.load(Ordering::SeqCst), 0);
        }

        #[derive(Clone)]
        struct InterruptingCompute {
            signal: InterruptSignal,
            calls: Arc<AtomicUsize>,
        }

        impl Compute for InterruptingCompute {
            type Input = u8;
            type Output = u8;

            fn computation_path(&self) -> &ComputationPath {
                &TEST_PATH
            }

            fn execute_with_encoded_input(
                &mut self,
                input: Self::Input,
                _encoded: &[u8],
            ) -> Result<Self::Output> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                self.signal.interrupt();
                Ok(input)
            }
        }

        #[test]
        fn signal_set_during_batch_skips_pending_work() {
            let signal = InterruptSignal::new();
            let calls = Arc::new(AtomicUsize::new(0));
            let compute = InterruptingCompute {
                signal: signal.clone(),
                calls: Arc::clone(&calls),
            };

            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("single-thread test pool should build");
            let results = pool.install(|| {
                compute
                    .with_inputs(0u8..4)
                    .with_interrupt_signal(signal)
                    .results()
            });

            assert_eq!(calls.load(Ordering::SeqCst), 1);
            assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
            assert_eq!(
                results
                    .iter()
                    .filter(|result| matches!(result, Err(Error::Interrupted)))
                    .count(),
                3
            );
        }
    }
}
