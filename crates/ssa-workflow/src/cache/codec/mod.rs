/// Stable, versioned name for an encoded cache value format.
///
/// `ValueFormat::concat` keeps composed formats as a static expression instead of materializing a
/// new `&'static str`. That avoids Rust's current restriction on using generic associated consts
/// in const-generic string concatenation while still letting codec wrappers expose a static
/// `CodecEngine::VALUE_FORMAT`.
#[derive(Clone, Copy, Debug)]
pub struct ValueFormat(ValueFormatRepr);

#[derive(Clone, Copy, Debug)]
enum ValueFormatRepr {
    Static(&'static str),
    Concat(&'static ValueFormat, &'static str),
}

impl ValueFormat {
    /// Create a value format identifier from a stable static name.
    pub const fn new(name: &'static str) -> Self {
        Self(ValueFormatRepr::Static(name))
    }

    /// Create a format by appending a static suffix to another static format expression.
    pub const fn concat(base: &'static ValueFormat, suffix: &'static str) -> Self {
        Self(ValueFormatRepr::Concat(base, suffix))
    }

    fn render_into(self, out: &mut String) {
        match self.0 {
            ValueFormatRepr::Static(value) => out.push_str(value),
            ValueFormatRepr::Concat(base, suffix) => {
                base.render_into(out);
                out.push_str(suffix);
            }
        }
    }

    /// Return the rendered value format identifier.
    pub fn render(self) -> String {
        let mut out = String::new();
        self.render_into(&mut out);
        out
    }
}

impl std::fmt::Display for ValueFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            ValueFormatRepr::Static(value) => f.write_str(value),
            ValueFormatRepr::Concat(base, suffix) => {
                std::fmt::Display::fmt(base, f)?;
                f.write_str(suffix)
            }
        }
    }
}

impl PartialEq for ValueFormat {
    fn eq(&self, other: &Self) -> bool {
        self.to_string() == other.to_string()
    }
}

impl Eq for ValueFormat {}

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

/// A codec engine that can serialize/deserialize `T`.
///
/// Engines can carry their own scratch space, so the same instance can be reused
/// across many calls without external buffer management:
///
/// ```rust
/// use ssa_workflow::{cache::codec::CodecEngine, prelude::*};
///
/// # #[cfg(feature = "bitcode06")]
/// # {
/// let value = 0u8;
/// let mut engine = Bitcode06::default();
/// let bytes = engine
///     .encode(&value)
///     .expect("bitcode always encodes this value")
///     .to_vec();
/// let value: u8 = engine.decode(&bytes).unwrap();
/// # }
/// ```
pub trait CodecEngine<T>: CloneFresh {
    /// Stable value format identifier for bytes produced by this codec type.
    const VALUE_FORMAT: ValueFormat;

    /// Encode `value` and return the encoded bytes, or `Err(SkipReason)` to skip caching.
    ///
    /// The returned slice borrows from the engine's internal buffer and is
    /// valid until the next call to `encode` or `decode` on this instance.
    fn encode(&mut self, value: &T) -> Result<&[u8], SkipReason>;

    /// Decode a `T` from `bytes`.
    fn decode(&mut self, bytes: &[u8]) -> Result<T, Error>;
}

type BoxedCodecError = Box<dyn std::error::Error + Send + Sync + 'static>;

/// Reasons why a value may be skipped from the cache.
#[derive(thiserror::Error, Debug)]
pub enum SkipReason {
    #[error("encoded value size {encoded_len} exceeds cache limit {max_len}")]
    EncodedValueTooLarge { encoded_len: usize, max_len: usize },

    #[error("codec could not encode value")]
    EncodeFailure(#[source] BoxedCodecError),
}

/// Errors produced by codec engines and codec adapters.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[cfg(feature = "bitcode06")]
    #[error("Bitcode codec error")]
    BitCode(#[from] bitcode::Error),

    #[cfg(feature = "postcard")]
    #[error("Postcard codec error")]
    Postcard(#[from] postcard::Error),

    #[error("Checked codec error")]
    Checked(#[from] crate::cache::codec::CheckedError),

    #[cfg(feature = "compress")]
    #[error("Compression codec error")]
    Compress(#[from] crate::cache::codec::compress::CompressError),

    #[cfg(test)]
    #[error("Fixture codec error")]
    Fixture(#[from] fixtures::Error),
}

impl Error {
    pub(crate) const fn is_cache_corruption(&self) -> bool {
        match self {
            Self::Checked(err) => err.is_cache_corruption(),
            #[cfg(feature = "compress")]
            Self::Compress(err) => err.is_cache_corruption(),
            #[cfg(any(feature = "bitcode06", feature = "postcard", test))]
            _ => false,
        }
    }
}

pub use checked::{CheckedCodec, Error as CheckedError};
#[cfg(feature = "bitcode06")]
pub use engine::bitcode::Bitcode06;
#[cfg(feature = "postcard")]
pub use engine::postcard::Postcard;

mod checked;
mod engine;

#[cfg(feature = "compress")]
pub mod compress;

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
pub mod fixtures;

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn checked_error_corruption_classification_is_exposed_at_codec_error_layer() {
        assert!(Error::Checked(CheckedError::TruncatedInput).is_cache_corruption());
        assert!(Error::Checked(CheckedError::ChecksumMismatch).is_cache_corruption());
    }

    #[test]
    fn value_format_concat_renders_and_compares_by_full_format() {
        const FORMAT: ValueFormat = ValueFormat::new("bitcode06/v1");
        const CHECKED: ValueFormat = ValueFormat::concat(&FORMAT, "+checked/v1");
        const COMPRESSED: ValueFormat = ValueFormat::concat(&CHECKED, "+zstd/v1");

        assert_eq!(CHECKED.render(), "bitcode06/v1+checked/v1");
        assert_eq!(CHECKED.to_string(), "bitcode06/v1+checked/v1");
        assert_eq!(
            COMPRESSED,
            ValueFormat::new("bitcode06/v1+checked/v1+zstd/v1")
        );
    }

    #[test]
    fn value_format_concat_can_be_built_at_runtime_from_static_parts() {
        static FORMAT: ValueFormat = ValueFormat::new("postcard/v1");

        let checked = ValueFormat::concat(&FORMAT, "+checked/v1");

        assert_eq!(checked.render(), "postcard/v1+checked/v1");
        assert_eq!(checked.to_string(), "postcard/v1+checked/v1");
    }

    #[cfg(feature = "compress")]
    #[test]
    fn compress_error_corruption_classification_is_exposed_at_codec_error_layer() {
        use crate::cache::codec::compress::CompressError;

        assert!(Error::Compress(CompressError::EmptyInput).is_cache_corruption());
        assert!(Error::Compress(CompressError::TruncatedInput).is_cache_corruption());
        assert!(Error::Compress(CompressError::ChecksumMismatch).is_cache_corruption());
        assert!(!Error::Compress(CompressError::UnsupportedVersion(0)).is_cache_corruption());
        assert!(
            !Error::Compress(CompressError::CompressionAlgorithmMismatch).is_cache_corruption()
        );
        assert!(!Error::Compress(CompressError::ContentTooLarge).is_cache_corruption());
    }
}
