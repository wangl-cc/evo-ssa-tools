#[cfg(feature = "checked")]
pub mod checked;
#[cfg(feature = "checked")]
pub use checked::CheckedCodec;

/// A codec engine that can encode/decode `T`
///
/// Engines can carry their own scratch space, so the same instance can be reused
/// across many calls without external buffer management:
///
/// ```rust
/// use ssa_pipeline::prelude::*;
///
/// let value = 0u8;
/// let mut engine = Bitcode::default();
/// let bytes = engine
///     .encode(&value)
///     .expect("bitcode always encodes this value")
///     .to_vec();
/// let value: u8 = engine.decode(&bytes).unwrap();
/// ```
pub trait CodecEngine<T> {
    /// Encode `value` and return the encoded bytes, or `Err(SkipReason)` to skip caching.
    ///
    /// The returned slice borrows from the engine's internal buffer and is
    /// valid until the next call to `encode` or `decode` on this instance.
    fn encode(&mut self, value: &T) -> Result<&[u8], SkipReason>;

    /// Decode a `T` from `bytes`.
    fn decode(&mut self, bytes: &[u8]) -> Result<T, Error>;
}

/// Factory for per-worker codec engines.
///
/// `execute_many` uses one engine instance per worker thread, created by calling this factory.
/// Factories may be stateful and can capture runtime configuration.
pub trait EngineFactory {
    type Engine;

    fn make_engine(&self) -> Self::Engine;
}

impl<F, E> EngineFactory for F
where
    F: Fn() -> E,
{
    type Engine = E;

    fn make_engine(&self) -> Self::Engine {
        (self)()
    }
}

/// Reasons why a value may be skipped from the cache.
#[derive(thiserror::Error, Debug)]
pub enum SkipReason {
    #[error("encoded value size {encoded_len} exceeds cache limit {max_len}")]
    EncodedValueTooLarge { encoded_len: usize, max_len: usize },
}

/// Errors produced by codec engines and codec adapters.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[cfg(feature = "bitcode")]
    #[error("Bitcode codec error")]
    BitCode(#[from] bitcode::Error),

    #[cfg(feature = "checked")]
    #[error("Checked codec error")]
    Checked(#[from] crate::cache::codec::checked::Error),

    #[cfg(feature = "compress")]
    #[error("Compression codec error")]
    Compress(#[from] crate::cache::codec::compress::frame::Error),

    #[cfg(test)]
    #[error("Fixture codec error")]
    Fixture(#[from] fixtures::Error),
}

impl Error {
    pub(crate) const fn is_cache_corruption(self: &Self) -> bool {
        #[allow(unreachable_patterns)]
        match self {
            #[cfg(feature = "checked")]
            Self::Checked(err) => err.is_cache_corruption(),
            #[cfg(feature = "compress")]
            Self::Compress(err) => err.is_cache_corruption(),
            _ => false,
        }
    }
}

pub mod engine;

#[cfg(feature = "compress")]
pub mod compress;

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
pub mod fixtures;
