use crate::Result;

/// Execution-facing cache abstraction keyed by canonical input bytes.
pub trait Cache<T> {
    /// Fetch a typed value by key, or execute and store it on cache miss.
    fn fetch_or_execute<F>(&mut self, key: &[u8], execute: F) -> Result<T>
    where
        F: FnOnce() -> Result<T>;
}

impl<T> Cache<T> for () {
    fn fetch_or_execute<F>(&mut self, _key: &[u8], execute: F) -> Result<T>
    where
        F: FnOnce() -> Result<T>,
    {
        execute()
    }
}

/// Produces a per-worker instance for parallel execution.
///
/// Called once per worker before dispatching work via `execute_many`. Each worker gets its
/// own instance so that per-call state is never shared across threads.
///
/// What `fork` actually does differs by type:
///
/// - **Caches** (`HashObjectCache`, `LruObjectCache`, `Fjall2Store`, `Fjall3Store`, `RedbStore`):
///   returns a new handle that shares the same underlying data via `Arc`. A result stored by one
///   worker is immediately visible to all others.
///
/// - **Codec engines** (`Bitcode06`, `Postcard`, `CheckedCodec`, `CompressedCodec`): creates a
///   fresh engine with the same settings but its own independent encode buffer. Workers never touch
///   each other's codec state.
///
/// `EncodedCache` combines both: its store side shares data across workers, its codec side is
/// per-worker.
pub trait Fork: Sized {
    fn fork(&self) -> Self;
}

impl Fork for () {
    fn fork(&self) -> Self {}
}

pub mod canonical_encode;
pub mod codec;
pub mod memory;
pub mod storage;

#[cfg(feature = "migrate")]
pub mod migrate;
