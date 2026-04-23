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

/// Clone a new handle that shares the same backing state.
///
/// This is the contract for caches and raw stores whose worker-local instance should keep pointing
/// at the same underlying data. Cloning is shallow at the logical level: one worker can populate
/// the cache or store and other workers immediately observe the same entries.
///
/// Typical implementations wrap an `Arc` or another shared database handle.
pub trait CloneShared: Sized {
    fn clone_shared(&self) -> Self;
}

impl CloneShared for () {
    fn clone_shared(&self) -> Self {}
}

/// Clone a fresh worker-local instance from the same configuration.
///
/// This is the contract for codecs and compression helpers whose worker-local instance should
/// reuse configuration while resetting scratch buffers, encode state, or compression contexts.
///
/// Cloning is fresh at the logical level: workers should not contend on shared mutable codec state.
pub trait CloneFresh: Sized {
    fn clone_fresh(&self) -> Self;
}

impl CloneFresh for () {
    fn clone_fresh(&self) -> Self {}
}

mod canonical_encode;
pub mod codec;
mod encoded;
pub mod memory;
pub mod storage;

#[cfg(feature = "migrate")]
pub mod migrate;
