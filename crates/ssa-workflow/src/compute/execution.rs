//! Core execution traits and batch execution.

use std::sync::{Arc, atomic};

use rayon::prelude::{
    FromParallelIterator, IndexedParallelIterator, IntoParallelIterator, ParallelIterator,
};

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
/// `Input` is encoded into canonical bytes via [`CanonicalEncode`]. Those bytes are the cache key.
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
    /// The buffer must have length at least `Self::Input::SIZE`.
    /// Implementations should only access `buffer[..Self::Input::SIZE]`.
    ///
    /// See [`CanonicalEncode`] for more details.
    unsafe fn execute_one_with_buffer(
        &mut self,
        input: Self::Input,
        encode_buffer: &mut [u8],
    ) -> Result<Self::Output> {
        // Safety: The safety is guaranteed by the caller.
        let encoded = unsafe { input.encode_with_buffer(encode_buffer) };
        self.execute_with_encoded_input(input, encoded)
    }

    /// Execute computation for one input.
    ///
    /// This is the safe, single-item counterpart to [`Self::execute_one_with_buffer`].
    ///
    /// If you need to execute multiple inputs, use [`Self::with_inputs`] instead.
    fn execute_one(&mut self, input: Self::Input) -> Result<Self::Output> {
        let mut encode_buffer = vec![0u8; Self::Input::SIZE];
        // Safety: The buffer is initialized with length Self::Input::SIZE.
        unsafe { self.execute_one_with_buffer(input, &mut encode_buffer) }
    }

    /// Create a batch execution for `inputs`.
    ///
    /// Parallel collection requires an [`IndexedParallelIterator`], while serial collection
    /// requires an [`IntoIterator`]. The selected execution method determines which input bounds
    /// are required.
    ///
    /// Unordered parallel inputs such as `HashSet` do not support parallel collection:
    ///
    /// ```compile_fail
    /// # use ssa_workflow::prelude::*;
    /// # use std::collections::HashSet;
    /// # fn main() -> ssa_workflow::Result<()> {
    /// let task = DeterministicTask::builder("unordered-inputs-v1")
    ///     .function(|input: u8| Ok(input))
    ///     .build()?;
    /// let inputs = HashSet::from([1u8, 2u8]);
    /// let _ = task
    ///     .with_inputs(inputs)
    ///     .collect::<ssa_workflow::Result<Vec<_>>>();
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
    {
        BatchExecution {
            compute: self,
            inputs,
            interrupt_signal: None,
            progress: None,
        }
    }
}

/// Lazy batch execution returned by [`Compute::with_inputs`].
pub struct BatchExecution<'a, C, I>
where
    C: Compute,
{
    compute: &'a C,
    inputs: I,
    interrupt_signal: Option<InterruptSignal>,
    progress: Option<BatchProgress>,
}

impl<'a, C, I> BatchExecution<'a, C, I>
where
    C: Compute,
{
    /// Interrupt pending work when `signal` is set.
    ///
    /// Items already executing are not forcibly cancelled.
    pub fn with_interrupt_signal(mut self, signal: InterruptSignal) -> Self {
        self.interrupt_signal = Some(signal);
        self
    }
}

impl<'a, C, I> BatchExecution<'a, C, I>
where
    C: Compute,
    I: IntoParallelIterator<Item = C::Input>,
    I::Iter: IndexedParallelIterator<Item = C::Input>,
{
    /// Enable item progress tracking for this batch.
    ///
    /// Progress is opt-in. The returned handle can be cloned and polled while the batch executes.
    /// `started` counts input items that have entered the computation, regardless of their result.
    /// `in_flight` counts items currently executing the computation.
    /// The number of finished attempts is `started - in_flight`; after fail-fast collection,
    /// `total - started` is the number of items that never entered the computation.
    ///
    /// Enabling progress converts the inputs into their indexed parallel iterator, so the tracked
    /// batch supports parallel collection through [`BatchExecution::collect`] or
    /// [`BatchExecution::collect_in`].
    ///
    /// # Panics
    ///
    /// Panics if the batch contains more than `2^48 - 1` items.
    pub fn track_progress(self) -> (BatchExecution<'a, C, I::Iter>, BatchProgress) {
        let inputs = self.inputs.into_par_iter();
        let progress = BatchProgress::new(inputs.len());
        let batch = BatchExecution {
            compute: self.compute,
            inputs,
            interrupt_signal: self.interrupt_signal,
            progress: Some(progress.clone()),
        };
        (batch, progress)
    }
}

impl<'a, C, I> BatchExecution<'a, C, I>
where
    C: Clone + Compute + Sync + 'a,
    I: IntoParallelIterator<Item = C::Input> + 'a,
    I::Iter: IndexedParallelIterator<Item = C::Input>,
    C::Output: Send,
{
    /// Convert this batch into its indexed parallel result iterator.
    ///
    /// Collecting this iterator as `Result<Vec<_>>` naturally short-circuits on errors.
    /// Collecting it as `Vec<Result<_>>` keeps successful and failed items.
    pub fn into_par_iter(self) -> impl IndexedParallelIterator<Item = Result<C::Output>> + 'a {
        let signal = self.interrupt_signal;
        let compute = self.compute;
        let progress = self.progress;
        self.inputs.into_par_iter().map_init(
            move || {
                (
                    vec![0u8; C::Input::SIZE],
                    compute.clone(),
                    signal.clone(),
                    progress.clone(),
                )
            },
            |(buffer, compute, signal, progress), input| {
                if let Some(signal) = signal
                    && signal.is_interrupted()
                {
                    return Err(Error::Interrupted);
                }

                let _in_flight = progress.as_ref().map(BatchProgress::begin_item);

                // Safety: The buffer is initialized with length Self::Input::SIZE.
                unsafe { compute.execute_one_with_buffer(input, buffer) }
            },
        )
    }

    /// Execute the batch in the current Rayon pool and collect into `T`.
    ///
    /// Outside a Rayon pool, execution uses the global pool. The result iterator follows input
    /// order; ordered targets such as `Vec` preserve that order.
    ///
    /// Collect as `Result<Vec<_>>` to short-circuit on errors, or as `Vec<Result<_>>` to retain
    /// every item result.
    pub fn collect<T>(self) -> T
    where
        T: FromParallelIterator<Result<C::Output>>,
    {
        self.into_par_iter().collect()
    }

    /// Execute the batch in `pool` and collect into `T`.
    ///
    /// The result iterator follows input order; ordered targets such as `Vec` preserve that order.
    ///
    /// Collect as `Result<Vec<_>>` to short-circuit on errors, or as `Vec<Result<_>>` to retain
    /// every item result.
    pub fn collect_in<T>(self, pool: &rayon::ThreadPool) -> T
    where
        T: FromParallelIterator<Result<C::Output>> + Send,
    {
        let execution = self.into_par_iter();
        pool.install(move || execution.collect())
    }
}

impl<'a, C, I> BatchExecution<'a, C, I>
where
    C: Clone + Compute + 'a,
    I: IntoIterator<Item = C::Input> + 'a,
{
    /// Execute the batch serially on the caller thread and collect into `T`.
    ///
    /// Results are produced in input order; ordered targets such as `Vec` preserve that order.
    ///
    /// Collect as `Result<Vec<_>>` to short-circuit on errors, or as `Vec<Result<_>>` to retain
    /// every item result.
    pub fn collect_serial<T>(self) -> T
    where
        T: FromIterator<Result<C::Output>>,
    {
        self.into_serial_iter().collect()
    }

    fn into_serial_iter(self) -> impl Iterator<Item = Result<C::Output>> + 'a {
        let signal = self.interrupt_signal;
        let mut compute = self.compute.clone();
        let mut buffer = vec![0u8; C::Input::SIZE];
        let progress = self.progress;

        self.inputs.into_iter().map(move |input| {
            if let Some(signal) = &signal
                && signal.is_interrupted()
            {
                return Err(Error::Interrupted);
            }

            let _in_flight = progress.as_ref().map(BatchProgress::begin_item);

            // Safety: The buffer is initialized with length Self::Input::SIZE.
            unsafe { compute.execute_one_with_buffer(input, &mut buffer) }
        })
    }
}

/// Shared item progress for one batch execution.
#[derive(Debug, Clone)]
pub struct BatchProgress {
    total: usize,
    state: Arc<PackedProgressState>,
}

impl BatchProgress {
    const MAX_TRACKED_ITEMS: usize = PackedProgressState::STARTED_MASK;

    fn new(total: usize) -> Self {
        assert!(
            total <= Self::MAX_TRACKED_ITEMS,
            "tracked batch size exceeds the 48-bit started-item capacity"
        );

        Self {
            total,
            state: Arc::new(PackedProgressState::default()),
        }
    }

    /// Return the number of input items in the tracked batch.
    pub const fn total(&self) -> usize {
        self.total
    }

    /// Return the number of input items that have entered the computation.
    pub fn started(&self) -> usize {
        self.state.snapshot().started
    }

    /// Return the number of input items currently executing the computation.
    pub fn in_flight(&self) -> usize {
        self.state.snapshot().in_flight
    }

    /// Return a consistent point-in-time progress snapshot.
    pub fn snapshot(&self) -> BatchProgressSnapshot {
        let state = self.state.snapshot();
        BatchProgressSnapshot {
            total: self.total,
            started: state.started,
            in_flight: state.in_flight,
        }
    }

    fn begin_item(&self) -> InFlightGuard<'_> {
        self.state.start();
        InFlightGuard { state: &self.state }
    }
}

#[derive(Debug, Default)]
struct PackedProgressState(atomic::AtomicUsize);

impl PackedProgressState {
    const IN_FLIGHT_BITS: u32 = 16;
    const IN_FLIGHT_MASK: usize = (1 << Self::IN_FLIGHT_BITS) - 1;
    const IN_FLIGHT_ONE: usize = 1 << Self::STARTED_BITS;
    const STARTED_BITS: u32 = 48;
    const STARTED_MASK: usize = Self::IN_FLIGHT_ONE - 1;

    fn start(&self) {
        let previous = self
            .0
            .fetch_add(Self::IN_FLIGHT_ONE + 1, atomic::Ordering::Relaxed);
        let previous = Self::decode(previous);
        debug_assert!(previous.in_flight < Self::IN_FLIGHT_MASK);
        debug_assert!(previous.started < Self::STARTED_MASK);
    }

    fn finish(&self) {
        let previous = self
            .0
            .fetch_sub(Self::IN_FLIGHT_ONE, atomic::Ordering::Relaxed);
        debug_assert!(Self::decode(previous).in_flight > 0);
    }

    fn snapshot(&self) -> PackedProgressSnapshot {
        Self::decode(self.0.load(atomic::Ordering::Relaxed))
    }

    const fn decode(state: usize) -> PackedProgressSnapshot {
        PackedProgressSnapshot {
            started: state & Self::STARTED_MASK,
            in_flight: state >> Self::STARTED_BITS,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PackedProgressSnapshot {
    started: usize,
    in_flight: usize,
}

struct InFlightGuard<'a> {
    state: &'a PackedProgressState,
}

impl Drop for InFlightGuard<'_> {
    fn drop(&mut self) {
        self.state.finish();
    }
}

/// Point-in-time item progress for one batch execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchProgressSnapshot {
    /// Number of input items in the batch.
    pub total: usize,
    /// Number of input items that have entered the computation.
    pub started: usize,
    /// Number of input items currently executing the computation.
    pub in_flight: usize,
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
            let mut buffer = vec![0xAA; u16::SIZE + 1];

            // Safety: the buffer is longer than the required u16 canonical encoding.
            let output = unsafe { compute.execute_one_with_buffer(0x1234, &mut buffer) }?;

            assert_eq!(output, [0x12, 0x34]);
            assert_eq!(buffer, [0x12, 0x34, 0xAA]);
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

        mod parallel {
            use std::{ops::Range, rc::Rc};

            use super::*;

            #[test]
            fn collect_into_vec_preserves_item_results() {
                let results = FallibleCompute
                    .with_inputs(0u8..4)
                    .collect::<Vec<Result<_>>>();

                assert_eq!(results.len(), 4);
                assert_eq!(results[0].as_ref().ok(), Some(&10));
                assert_eq!(results[1].as_ref().ok(), Some(&11));
                assert!(matches!(results[2], Err(Error::Compute(_))));
                assert_eq!(results[3].as_ref().ok(), Some(&13));
            }

            #[derive(Clone)]
            struct ThreadIndexCompute;

            impl Compute for ThreadIndexCompute {
                type Input = u8;
                type Output = Option<usize>;

                fn computation_path(&self) -> &ComputationPath {
                    &TEST_PATH
                }

                fn execute_with_encoded_input(
                    &mut self,
                    _input: Self::Input,
                    _encoded: &[u8],
                ) -> Result<Self::Output> {
                    Ok(rayon::current_thread_index())
                }
            }

            #[test]
            fn collect_in_uses_specified_pool() -> Result<()> {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(1)
                    .build()
                    .expect("single-thread test pool should build");
                let outputs = ThreadIndexCompute
                    .with_inputs(0u8..8)
                    .collect_in::<Result<Vec<_>>>(&pool)?;

                assert_eq!(outputs, vec![Some(0); 8]);
                Ok(())
            }

            struct LocalParallelInputs {
                range: Range<u8>,
                marker: Rc<()>,
            }

            impl IntoParallelIterator for LocalParallelInputs {
                type Item = u8;
                type Iter = <Range<u8> as IntoParallelIterator>::Iter;

                fn into_par_iter(self) -> Self::Iter {
                    drop(self.marker);
                    self.range.into_par_iter()
                }
            }

            #[test]
            fn collect_in_accepts_local_sources_with_send_parallel_iterators() -> Result<()> {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(1)
                    .build()
                    .expect("single-thread test pool should build");
                let inputs = LocalParallelInputs {
                    range: 0..2,
                    marker: Rc::new(()),
                };

                let outputs = FallibleCompute
                    .with_inputs(inputs)
                    .collect_in::<Result<Vec<_>>>(&pool)?;

                assert_eq!(outputs, [10, 11]);
                Ok(())
            }
        }

        mod serial {
            use super::*;

            #[test]
            fn collect_into_result_preserves_order() -> Result<()> {
                let outputs = FallibleCompute
                    .with_inputs([3u8, 1, 0])
                    .collect_serial::<Result<Vec<_>>>()?;

                assert_eq!(outputs, [13, 11, 10]);
                Ok(())
            }

            #[test]
            fn collect_into_vec_preserves_order_and_item_results() {
                let results = FallibleCompute
                    .with_inputs(0u8..4)
                    .collect_serial::<Vec<Result<_>>>();

                assert_eq!(results.len(), 4);
                assert_eq!(results[0].as_ref().ok(), Some(&10));
                assert_eq!(results[1].as_ref().ok(), Some(&11));
                assert!(matches!(results[2], Err(Error::Compute(_))));
                assert_eq!(results[3].as_ref().ok(), Some(&13));
            }

            #[derive(Clone)]
            struct CountingFallibleCompute {
                calls: Arc<AtomicUsize>,
            }

            impl Compute for CountingFallibleCompute {
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
                    if input == 2 {
                        Err(Error::Compute(Box::new(TestComputeError)))
                    } else {
                        Ok(input)
                    }
                }
            }

            #[test]
            fn collect_into_result_stops_on_first_error() {
                let calls = Arc::new(AtomicUsize::new(0));
                let result = CountingFallibleCompute {
                    calls: Arc::clone(&calls),
                }
                .with_inputs(0u8..5)
                .collect_serial::<Result<Vec<_>>>();

                assert!(matches!(result, Err(Error::Compute(_))));
                assert_eq!(calls.load(Ordering::SeqCst), 3);
            }
        }
    }

    mod progress_tracking {
        use super::*;

        #[derive(Clone)]
        struct CountingCompute;

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
                Ok(input)
            }
        }

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
                    Ok(input)
                }
            }
        }

        #[test]
        fn track_progress_reports_batch_total_before_execution() {
            let (_batch, progress) = CountingCompute.with_inputs(0u8..4).track_progress();

            assert_eq!(progress.total(), 4);
            assert_eq!(progress.snapshot(), BatchProgressSnapshot {
                total: 4,
                started: 0,
                in_flight: 0,
            });
        }

        #[test]
        fn track_progress_counts_started_items() {
            let (batch, progress) = CountingCompute.with_inputs(0u8..4).track_progress();

            let results = batch.collect::<Vec<Result<_>>>();

            assert_eq!(results.len(), 4);
            assert_eq!(progress.total(), 4);
            assert_eq!(progress.started(), 4);
            assert_eq!(progress.in_flight(), 0);
            assert_eq!(progress.snapshot(), BatchProgressSnapshot {
                total: 4,
                started: 4,
                in_flight: 0,
            });
        }

        #[test]
        fn track_progress_counts_failed_items_as_started() {
            let (batch, progress) = FallibleCompute.with_inputs(0u8..4).track_progress();

            let results = batch.collect::<Vec<Result<_>>>();

            assert_eq!(results.len(), 4);
            assert!(matches!(results[2], Err(Error::Compute(_))));
            assert_eq!(progress.snapshot(), BatchProgressSnapshot {
                total: 4,
                started: 4,
                in_flight: 0,
            });
        }

        #[derive(Clone)]
        struct BlockingCompute {
            entered: Arc<(std::sync::Mutex<usize>, std::sync::Condvar)>,
            release: Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>,
        }

        impl Compute for BlockingCompute {
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
                let (entered_lock, entered_condvar) = &*self.entered;
                let mut entered_count = entered_lock
                    .lock()
                    .expect("entered count mutex should not be poisoned");
                *entered_count += 1;
                entered_condvar.notify_all();
                drop(entered_count);

                let (release_lock, release_condvar) = &*self.release;
                let mut released = release_lock
                    .lock()
                    .expect("release mutex should not be poisoned");
                while !*released {
                    released = release_condvar
                        .wait(released)
                        .expect("release mutex should not be poisoned");
                }

                Ok(input)
            }
        }

        #[test]
        fn track_progress_reports_in_flight_item_during_execution() {
            let entered = Arc::new((std::sync::Mutex::new(0usize), std::sync::Condvar::new()));
            let release = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
            let compute = BlockingCompute {
                entered: Arc::clone(&entered),
                release: Arc::clone(&release),
            };
            let (batch, progress) = compute.with_inputs(0u8..1).track_progress();
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("single-thread test pool should build");

            std::thread::scope(|scope| {
                let handle = scope.spawn(|| pool.install(|| batch.collect::<Vec<Result<_>>>()));

                let (entered_lock, entered_condvar) = &*entered;
                let mut entered_count = entered_lock
                    .lock()
                    .expect("entered count mutex should not be poisoned");
                while *entered_count == 0 {
                    entered_count = entered_condvar
                        .wait(entered_count)
                        .expect("entered count mutex should not be poisoned");
                }
                drop(entered_count);

                assert_eq!(progress.snapshot(), BatchProgressSnapshot {
                    total: 1,
                    started: 1,
                    in_flight: 1,
                });

                let (release_lock, release_condvar) = &*release;
                let mut released = release_lock
                    .lock()
                    .expect("release mutex should not be poisoned");
                *released = true;
                release_condvar.notify_all();
                drop(released);

                let results = handle.join().expect("batch worker should not panic");
                assert_eq!(results.len(), 1);
            });

            assert_eq!(progress.snapshot(), BatchProgressSnapshot {
                total: 1,
                started: 1,
                in_flight: 0,
            });
        }

        #[derive(Clone)]
        struct PanickingCompute;

        impl Compute for PanickingCompute {
            type Input = u8;
            type Output = u8;

            fn computation_path(&self) -> &ComputationPath {
                &TEST_PATH
            }

            fn execute_with_encoded_input(
                &mut self,
                _input: Self::Input,
                _encoded: &[u8],
            ) -> Result<Self::Output> {
                panic!("test compute panic");
            }
        }

        #[test]
        fn panicking_item_leaves_no_in_flight_progress() {
            let (batch, progress) = PanickingCompute.with_inputs(0u8..1).track_progress();

            let panic = std::panic::catch_unwind(|| batch.collect::<Vec<Result<_>>>());

            assert!(panic.is_err());
            assert_eq!(progress.snapshot(), BatchProgressSnapshot {
                total: 1,
                started: 1,
                in_flight: 0,
            });
        }

        #[test]
        fn packed_progress_state_updates_counts_atomically() {
            let state = PackedProgressState::default();

            state.start();
            state.start();
            assert_eq!(state.snapshot(), PackedProgressSnapshot {
                started: 2,
                in_flight: 2,
            });

            state.finish();
            assert_eq!(state.snapshot(), PackedProgressSnapshot {
                started: 2,
                in_flight: 1,
            });

            state.finish();
            assert_eq!(state.snapshot(), PackedProgressSnapshot {
                started: 2,
                in_flight: 0,
            });
        }

        #[test]
        fn packed_progress_state_reaches_started_capacity_without_carrying() {
            let initial = PackedProgressState::STARTED_MASK - 1;
            let state = PackedProgressState(atomic::AtomicUsize::new(initial));

            state.start();
            assert_eq!(state.snapshot(), PackedProgressSnapshot {
                started: PackedProgressState::STARTED_MASK,
                in_flight: 1,
            });

            state.finish();

            assert_eq!(state.snapshot(), PackedProgressSnapshot {
                started: PackedProgressState::STARTED_MASK,
                in_flight: 0,
            });
        }

        #[test]
        fn packed_in_flight_capacity_covers_rayon_thread_limit() {
            assert!(
                rayon::max_num_threads() <= PackedProgressState::IN_FLIGHT_MASK,
                "packed progress must represent every worker in one Rayon pool"
            );
        }

        #[test]
        #[should_panic(expected = "tracked batch size exceeds the 48-bit started-item capacity")]
        fn progress_rejects_batch_larger_than_started_capacity() {
            let _ = BatchProgress::new(PackedProgressState::STARTED_MASK + 1);
        }

        #[test]
        fn interrupted_items_are_not_started_or_in_flight() {
            let signal = InterruptSignal::new();
            signal.interrupt();
            let (batch, progress) = CountingCompute
                .with_inputs(0u8..4)
                .with_interrupt_signal(signal)
                .track_progress();

            let results = batch.collect::<Vec<Result<_>>>();

            assert!(
                results
                    .iter()
                    .all(|result| matches!(result, Err(Error::Interrupted)))
            );
            assert_eq!(progress.snapshot(), BatchProgressSnapshot {
                total: 4,
                started: 0,
                in_flight: 0,
            });
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

        mod parallel {
            use super::*;

            #[test]
            fn collect_into_vec_preserves_preset_interrupts() {
                let signal = InterruptSignal::new();
                signal.interrupt();
                let calls = Arc::new(AtomicUsize::new(0));

                let results = CountingCompute {
                    calls: Arc::clone(&calls),
                }
                .with_inputs(0u8..4)
                .with_interrupt_signal(signal)
                .collect::<Vec<Result<_>>>();

                assert_eq!(calls.load(Ordering::SeqCst), 0);
                assert_eq!(results.len(), 4);
                assert!(
                    results
                        .iter()
                        .all(|result| { matches!(result, Err(Error::Interrupted)) })
                );
            }

            #[test]
            fn collect_into_result_observes_preset_interrupt() {
                let signal = InterruptSignal::new();
                signal.interrupt();
                let calls = Arc::new(AtomicUsize::new(0));

                let result = CountingCompute {
                    calls: Arc::clone(&calls),
                }
                .with_inputs(0u8..4)
                .with_interrupt_signal(signal)
                .collect::<Result<Vec<_>>>();

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
            fn collect_in_skips_pending_work_after_interrupt() {
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
                let results = compute
                    .with_inputs(0u8..4)
                    .with_interrupt_signal(signal)
                    .collect_in::<Vec<Result<_>>>(&pool);

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

        mod serial {
            use super::*;

            #[test]
            fn collect_into_vec_preserves_preset_interrupts() {
                let signal = InterruptSignal::new();
                signal.interrupt();
                let calls = Arc::new(AtomicUsize::new(0));

                let results = CountingCompute {
                    calls: Arc::clone(&calls),
                }
                .with_inputs(0u8..4)
                .with_interrupt_signal(signal)
                .collect_serial::<Vec<Result<_>>>();

                assert_eq!(calls.load(Ordering::SeqCst), 0);
                assert_eq!(results.len(), 4);
                assert!(
                    results
                        .iter()
                        .all(|result| matches!(result, Err(Error::Interrupted)))
                );
            }
        }
    }
}
