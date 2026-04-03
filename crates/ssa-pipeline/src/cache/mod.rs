pub use canonical_encode::{CanonicalEncode, CanonicalEncodeWriter};
pub use encoded::EncodedCache;

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

/// Produces a per-worker execution instance.
///
/// Called once per worker before dispatching work via `execute_many`.
///
/// `fork` does not necessarily mean "deep copy everything". It means "produce the worker-local
/// execution state this type needs". Depending on the type, that worker-local state may still
/// point at shared backing data.
///
/// Typical patterns:
///
/// - **Object caches / raw stores** (`HashObjectCache`, `LruObjectCache`, `Fjall2Store`,
///   `Fjall3Store`, `RedbStore`): return a fresh handle that still shares the same underlying
///   storage via `Arc`. Values written by one worker are immediately visible to others.
///
/// - **Codec engines** (`Bitcode06`, `Postcard`, `CheckedCodec`, `CompressedCodec`): create a
///   fresh engine with the same configuration but independent scratch space. Workers never touch
///   each other's codec buffers.
///
/// `EncodedCache` combines both: its store side shares backing storage across workers, while its
/// codec side is worker-local.
pub trait Fork: Sized {
    fn fork(&self) -> Self;
}

impl Fork for () {
    fn fork(&self) -> Self {}
}

mod canonical_encode;
pub mod codec;
mod encoded;
pub mod memory;
pub mod storage;

#[cfg(feature = "migrate")]
pub mod migrate;
