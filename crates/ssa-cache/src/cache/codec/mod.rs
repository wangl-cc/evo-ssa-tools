use crate::Result;

/// A codec engine that can encode/decode `T`
///
/// Engines can carry their own scratch space, so the same instance can be reused
/// across many calls without external buffer management:
///
/// ```rust
/// use ssa_cache::prelude::*;
///
/// let value = 0u8;
/// let mut engine = Bitcode::default();
/// let bytes = engine
///     .encode(&value)
///     .expect("bitcode never skips cache")
///     .to_vec();
/// let value: u8 = engine.decode(&bytes).unwrap();
/// ```
///
/// All engine types must implement [`Default`] (to construct instances for each worker).
pub trait CodecEngine<T>: Default {
    /// Encode `value` and return the encoded bytes, or `Err(SkipReason)` to skip caching.
    ///
    /// The returned slice borrows from the engine's internal buffer and is
    /// valid until the next call to `encode` or `decode` on this instance.
    fn encode(&mut self, value: &T) -> Result<&[u8], SkipReason>;

    /// Decode a `T` from `bytes`.
    fn decode(&mut self, bytes: &[u8]) -> Result<T>;
}

#[derive(thiserror::Error, Debug)]
pub enum SkipReason {
    #[error("encoded value size {encoded_len} exceeds cache limit {max_len}")]
    EncodedValueTooLarge { encoded_len: usize, max_len: usize },
}

#[cfg(feature = "bitcode")]
pub mod bitcode;

#[cfg(feature = "compress")]
pub mod compress;
