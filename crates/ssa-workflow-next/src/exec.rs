use std::{
    num::NonZeroUsize,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
    thread,
};

use parking_lot::Mutex;

use crate::{Error, Result};

/// Execution resources for one request: how many worker threads compute its missing items.
///
/// Each materialization starts that many scoped threads (fewer if there are fewer items). Workers
/// claim item indices from a shared counter, so at most `threads` items are in flight, and each
/// worker holds exactly one worker state. There is no work stealing: a worker never starts a
/// second item while one is unfinished, so a kernel may use its own thread pool (Rayon or
/// otherwise) without nesting into ours.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Exec {
    threads: NonZeroUsize,
}

impl Exec {
    /// Run on `threads` worker threads (at least one).
    pub fn threads(threads: usize) -> Self {
        Self {
            threads: NonZeroUsize::new(threads).unwrap_or(NonZeroUsize::MIN),
        }
    }

    /// Run on the calling thread.
    pub fn serial() -> Self {
        Self::threads(1)
    }

    /// Number of worker threads.
    pub fn thread_count(&self) -> usize {
        self.threads.get()
    }

    /// Compute `f(state, i)` for every `i` in `0..len`, returning the results in index order.
    ///
    /// Each worker checks out one state when it starts and checks it back in when it stops. The
    /// first error stops workers from claiming new items; a panic in a worker is resumed on the
    /// caller after the other workers stop.
    pub(crate) fn map<S, T>(
        &self,
        len: usize,
        checkout: impl Fn() -> S + Sync,
        checkin: impl Fn(S) + Sync,
        f: impl Fn(&mut S, usize) -> Result<T> + Sync,
    ) -> Result<Vec<T>>
    where
        T: Send,
    {
        if len == 0 {
            return Ok(Vec::new());
        }
        let workers = self.threads.get().min(len);
        let next = AtomicUsize::new(0);
        let stop = AtomicBool::new(false);
        let error: Mutex<Option<(usize, Error)>> = Mutex::new(None);

        let work = || {
            let _stop_on_panic = StopOnPanic(&stop);
            let mut state = checkout();
            let mut done = Vec::new();
            while !stop.load(Ordering::Relaxed) {
                let index = next.fetch_add(1, Ordering::Relaxed);
                if index >= len {
                    break;
                }
                match f(&mut state, index) {
                    Ok(value) => done.push((index, value)),
                    Err(err) => {
                        stop.store(true, Ordering::Relaxed);
                        let mut slot = error.lock();
                        // Report the lowest failing index, so the error does not depend on
                        // scheduling when several items fail.
                        if slot.as_ref().is_none_or(|(first, _)| index < *first) {
                            *slot = Some((index, err));
                        }
                        break;
                    }
                }
            }
            checkin(state);
            done
        };

        let parts: Vec<Vec<(usize, T)>> = if workers == 1 {
            vec![work()]
        } else {
            thread::scope(|scope| {
                let handles: Vec<_> = (0..workers).map(|_| scope.spawn(work)).collect();
                let mut parts = Vec::with_capacity(workers);
                let mut panic = None;
                for handle in handles {
                    match handle.join() {
                        Ok(part) => parts.push(part),
                        Err(payload) => panic = panic.or(Some(payload)),
                    }
                }
                if let Some(payload) = panic {
                    std::panic::resume_unwind(payload);
                }
                parts
            })
        };

        if let Some((_, err)) = error.into_inner() {
            return Err(err);
        }
        let mut slots: Vec<Option<T>> = std::iter::repeat_with(|| None).take(len).collect();
        for (index, value) in parts.into_iter().flatten() {
            slots[index] = Some(value);
        }
        Ok(slots
            .into_iter()
            .map(|value| value.expect("every index is claimed once when nothing failed"))
            .collect())
    }
}

impl Default for Exec {
    /// One worker per available CPU.
    fn default() -> Self {
        Self {
            threads: thread::available_parallelism().unwrap_or(NonZeroUsize::MIN),
        }
    }
}

struct StopOnPanic<'a>(&'a AtomicBool);

impl Drop for StopOnPanic<'_> {
    fn drop(&mut self) {
        if thread::panicking() {
            self.0.store(true, Ordering::Relaxed);
        }
    }
}
