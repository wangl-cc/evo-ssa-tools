//! Serialisation engines and format identifiers for persistent cache values.
//!
//! # Architecture
//!
//! The codec layer sits between typed Rust values and raw storage backends. It has three
//! responsibilities:
//!
//! - **Serialisation**: convert `T` to bytes via [`CodecEngine::encode`].
//! - **Deserialisation**: recover `T` from bytes via [`CodecEngine::decode`].
//! - **Format identity**: tag every stored value with a stable, versioned name so that different
//!   serialisation formats, compression choices, or schema versions map to distinct cache
//!   namespaces.
//!
//! # Key types
//!
//! | Type | Role |
//! |---|---|
//! | [`CodecEngine<T>`] | The central trait: one `encode` / `decode` pair per type `T`. |
//! | [`ValueFormat`] | A tree of `&'static str` segments that identifies the codec pipeline. |
//! | [`CheckedCodec<E>`] | Wraps any engine with a CRC32C integrity checksum. |
//! | [`CompressedCodec<E, C, P>`] | Wraps any engine with a framed compression layer (requires `compress` feature). |
//! | [`CloneFresh`] | Per-worker clone contract: shared config, fresh scratch buffers. |
//!
//! # Naming convention
//!
//! Format names use `-` for internal word separation and **never** contain `/`.
//! The `/` separator is reserved for layer composition via [`ValueFormat::concat`].
//! Good names: `"bitcode06-v1"`, `"postcard-v1"`, `"checked-v1"`, `"lz4-v1"`.
//!
//! # Composing codecs
//!
//! Codec wrappers nest via [`ValueFormat::concat`], which inserts `/` between layers:
//!
//! ```ignore
//! // Renders as "bitcode06-v1/checked-v1"
//! const VALUE_FORMAT: ValueFormat =
//!     ValueFormat::concat(&Bitcode06::<MyType>::VALUE_FORMAT, "checked-v1");
//! ```
//!
//! This structural naming ensures that changing any layer in the pipeline (e.g. switching from
//! `bitcode` to `postcard`, or adding compression) automatically selects a different persistent
//! namespace.
//!
//! # Built-in engines
//!
//! | Engine | Feature | Notes |
//! |---|---|---|
//! | [`Bitcode06`] | `bitcode06` | Fast binary format; tied to `bitcode` v0.6 major. |
//! | [`Postcard`] | `postcard` | Stable serde-based format with a published wire spec. |
//! | [`CheckedCodec<E>`] | always | CRC32C wrapper for any engine. |
//! | [`CompressedCodec<E, C, P>`] | `compress` | Framed compression (Lz4 / Zstd) for any engine. |

use std::hash::{Hash, Hasher};

/// Stable, versioned name for an encoded cache value format.
///
/// `ValueFormat::concat` keeps composed formats as a static expression instead of materializing a
/// new `&'static str`. That avoids Rust's current restriction on using generic associated consts
/// in const-generic string concatenation while still letting codec wrappers expose a static
/// `CodecEngine::VALUE_FORMAT`.
///
/// See the [module-level naming convention](self#naming-convention) for segment naming
/// rules.
///
/// # Equality
///
/// Equality is structural: two formats are equal only when they were built from the same chain of
/// [`concat`](Self::concat) calls. Segment boundaries carry semantic meaning (each suffix
/// represents a wrapper or adaptation layer).
#[derive(Clone, Copy, Debug)]
pub struct ValueFormat(ValueFormatRepr);

#[derive(Clone, Copy, Debug)]
enum ValueFormatRepr {
    Static(&'static str),
    Concat(&'static ValueFormat, &'static str),
}

use ValueFormatRepr::{Concat, Static};

impl ValueFormat {
    /// Create a value format identifier from a stable static name.
    pub const fn new(name: &'static str) -> Self {
        Self(Static(name))
    }

    /// Create a format by appending a static suffix to another static format expression.
    ///
    /// The rendered form inserts `/` between segments: `base/suffix`.
    /// Callers should **not** include the separator in `suffix`.
    pub const fn concat(base: &'static ValueFormat, suffix: &'static str) -> Self {
        Self(Concat(base, suffix))
    }
}

impl std::fmt::Display for ValueFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            Static(value) => f.write_str(value),
            Concat(base, suffix) => {
                std::fmt::Display::fmt(base, f)?;
                f.write_str("/")?;
                f.write_str(suffix)
            }
        }
    }
}

impl Hash for ValueFormat {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self.0 {
            Static(s) => {
                state.write_u8(0);
                state.write(s.as_bytes());
            }
            Concat(base, suffix) => {
                state.write_u8(1);
                Hash::hash(base, state);
                state.write(suffix.as_bytes());
            }
        }
    }
}

impl PartialEq for ValueFormat {
    fn eq(&self, other: &Self) -> bool {
        core::ptr::eq(self, other) || self.0 == other.0
    }
}

impl Eq for ValueFormat {}

impl PartialEq for ValueFormatRepr {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Static(a), Static(b)) => a == b,
            (Concat(a_base, a_suffix), Concat(b_base, b_suffix)) => {
                a_suffix == b_suffix && a_base.0 == b_base.0
            }
            _ => false,
        }
    }
}

// ---------------------------------------------------------------------------
// Traits
// ---------------------------------------------------------------------------

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
    ///
    /// Follow the [module naming convention](self#naming-convention): use `-` internally,
    /// never `/`. For wrappers, compose with [`ValueFormat::concat`]:
    ///
    /// ```ignore
    /// const VALUE_FORMAT: ValueFormat =
    ///     ValueFormat::concat(&InnerEngine::VALUE_FORMAT, "my-wrapper-v1");
    /// ```
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
    fn concat_renders_with_slash_separator() {
        const FORMAT: ValueFormat = ValueFormat::new("bitcode06-v1");
        const CHECKED: ValueFormat = ValueFormat::concat(&FORMAT, "checked-v1");
        const COMPRESSED: ValueFormat = ValueFormat::concat(&CHECKED, "zstd-v1");

        assert_eq!(CHECKED.to_string(), "bitcode06-v1/checked-v1");
        assert_eq!(CHECKED.to_string(), "bitcode06-v1/checked-v1");
        assert_eq!(COMPRESSED.to_string(), "bitcode06-v1/checked-v1/zstd-v1");
    }

    #[test]
    fn concat_with_same_chain_is_equal() {
        const BASE: ValueFormat = ValueFormat::new("bitcode06-v1");
        const A: ValueFormat = ValueFormat::concat(&BASE, "checked-v1");
        const B: ValueFormat = ValueFormat::concat(&BASE, "checked-v1");

        assert_eq!(A, B);
    }

    #[test]
    fn static_never_equals_concat() {
        const BASE: ValueFormat = ValueFormat::new("bitcode06-v1");
        let concat = ValueFormat::concat(&BASE, "checked-v1");

        // Same rendered bytes, different structure.
        assert_ne!(ValueFormat::new("bitcode06-v1/checked-v1"), concat);
        assert_ne!(concat, ValueFormat::new("bitcode06-v1/checked-v1"));
    }

    #[test]
    fn hash_distinguishes_static_from_concat() {
        const BASE: ValueFormat = ValueFormat::new("bitcode06-v1");
        let concat = ValueFormat::concat(&BASE, "checked-v1");
        let static_eq = ValueFormat::new("bitcode06-v1/checked-v1");

        use std::collections::hash_map::DefaultHasher;
        let hash = |vf: ValueFormat| {
            let mut h = DefaultHasher::new();
            vf.hash(&mut h);
            h.finish()
        };

        assert_ne!(hash(concat), hash(static_eq));
    }

    #[test]
    fn concat_can_be_built_at_runtime_from_static_parts() {
        static FORMAT: ValueFormat = ValueFormat::new("postcard-v1");

        let checked = ValueFormat::concat(&FORMAT, "checked-v1");

        assert_eq!(checked.to_string(), "postcard-v1/checked-v1");
        assert_eq!(checked.to_string(), "postcard-v1/checked-v1");
    }

    #[test]
    fn eq_static_only_matches_same_string() {
        assert_eq!(ValueFormat::new("alpha-v1"), ValueFormat::new("alpha-v1"));
        assert_ne!(ValueFormat::new("alpha-v1"), ValueFormat::new("beta-v1"));
    }

    #[test]
    fn eq_concat_only_matches_same_chain() {
        const BASE: ValueFormat = ValueFormat::new("base-v1");
        let a = ValueFormat::concat(&BASE, "x-v1");
        let b = ValueFormat::concat(&BASE, "x-v1");

        assert_eq!(a, b);
        assert_ne!(a, ValueFormat::concat(&BASE, "y-v1"));
    }

    #[test]
    fn eq_deeply_nested_concat_compares_recursively() {
        const A: ValueFormat = ValueFormat::new("a");
        const AB: ValueFormat = ValueFormat::concat(&A, "b");
        const ABC: ValueFormat = ValueFormat::concat(&AB, "c");

        const A2: ValueFormat = ValueFormat::new("a");
        const AB2: ValueFormat = ValueFormat::concat(&A2, "b");
        const ABC2: ValueFormat = ValueFormat::concat(&AB2, "c");

        assert_eq!(ABC, ABC2);
        assert_ne!(ABC, ValueFormat::concat(&AB, "d"));
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
