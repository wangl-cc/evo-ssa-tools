//! Codecs for turning typed cache values into bytes.
//!
//! A codec has two jobs:
//!
//! - encode and decode one Rust value type through [`CodecEngine`]
//! - expose a stable [`ValueFormat`] name for the bytes it writes
//!
//! Persistent cache namespaces include the value format. Changing the serializer, adding a wrapper
//! such as [`CheckedCodec`], or changing a compression algorithm therefore moves new writes into a
//! different namespace instead of mixing incompatible bytes.
//!
//! # Built-in codecs
//!
//! | Codec | Feature | Use when |
//! |---|---|---|
//! | [`Bitcode06`] | `bitcode06` | You want a fast binary format tied to `bitcode` 0.6. |
//! | [`Postcard`] | `postcard` | You want a compact serde format with a published wire spec. |
//! | [`CheckedCodec<E>`] | always | You want CRC32C corruption checks around another codec. |
//! | [`CompressedCodec<E, C, P>`] | `compress` | You want framed compression around another codec. |
//!
//! # Value format names
//!
//! Format names are part of the persistent cache contract. Use stable, versioned names such as
//! `"bitcode06-v1"`, `"postcard-v1"`, `"checked-v1"`, or `"lz4-v1"`.
//!
//! Use `-` inside one segment and never include `/`. The `/` separator is reserved for composed
//! formats built with [`ValueFormat::concat`].
//!
//! Codec wrappers append their own segment to the inner format:
//!
//! ```ignore
//! // Renders as "bitcode06-v1/checked-v1"
//! const VALUE_FORMAT: ValueFormat =
//!     ValueFormat::concat(&Bitcode06::<MyType>::VALUE_FORMAT, "checked-v1");
//! ```
//!
//! Structural names distinguish `ValueFormat::new("a/b")` from
//! `ValueFormat::concat(&ValueFormat::new("a"), "b")`, even though both display as `a/b`.
//!
//! [`CompressedCodec<E, C, P>`]: compress::CompressedCodec

use std::hash::{Hash, Hasher};

/// Stable name for bytes written by a codec pipeline.
///
/// Persistent cache namespaces include this value, so changing it intentionally separates old and
/// new cache entries. Keep names stable for compatible bytes and use a new versioned segment when
/// the encoded representation changes.
///
/// See the [module-level naming convention](self#naming-convention) for segment naming
/// rules.
///
/// # Equality
///
/// Equality is structural: segment boundaries matter. A single segment named `"a/b"` is different
/// from `"a"` composed with suffix `"b"` through [`concat`](Self::concat), even though both render
/// as `a/b`.
#[derive(Clone, Copy, Debug)]
pub struct ValueFormat(ValueFormatRepr);

#[derive(Clone, Copy, Debug)]
enum ValueFormatRepr {
    Static(&'static str),
    Concat(&'static ValueFormat, &'static str),
}

use ValueFormatRepr::{Concat, Static};

impl ValueFormat {
    /// Create a value format from one stable segment.
    pub const fn new(name: &'static str) -> Self {
        Self(Static(name))
    }

    /// Append one stable segment to an existing format.
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

/// Clone a worker-local instance from the same configuration.
///
/// Implementations should copy configuration, but not share mutable scratch buffers or codec
/// contexts. This lets parallel workers reuse the same codec settings without contending on codec
/// internals.
pub trait CloneFresh: Sized {
    /// Return an independent instance with the same configuration.
    fn clone_fresh(&self) -> Self;
}

impl CloneFresh for () {
    fn clone_fresh(&self) -> Self {}
}

/// Encode and decode one cache value type.
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
    /// Stable format name for bytes produced by this codec.
    ///
    /// Follow the [module naming convention](self#naming-convention): use `-` internally,
    /// never `/`. For wrappers, compose with [`ValueFormat::concat`]:
    ///
    /// ```ignore
    /// const VALUE_FORMAT: ValueFormat =
    ///     ValueFormat::concat(&InnerEngine::VALUE_FORMAT, "my-wrapper-v1");
    /// ```
    const VALUE_FORMAT: ValueFormat;

    /// Encode `value`, or return [`SkipReason`] to skip this cache write.
    ///
    /// The returned slice borrows from the engine's internal buffer and is
    /// valid until the next call to `encode` or `decode` on this instance.
    fn encode(&mut self, value: &T) -> Result<&[u8], SkipReason>;

    /// Decode one value from bytes.
    fn decode(&mut self, bytes: &[u8]) -> Result<T, Error>;
}

type BoxedCodecError = Box<dyn std::error::Error + Send + Sync + 'static>;

/// Reasons why a cache write may be skipped.
#[derive(thiserror::Error, Debug)]
pub enum SkipReason {
    #[error("encoded value size {encoded_len} exceeds cache limit {max_len}")]
    EncodedValueTooLarge { encoded_len: usize, max_len: usize },

    #[error("codec could not encode value")]
    EncodeFailure(#[source] BoxedCodecError),
}

/// Errors produced while decoding cached bytes.
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
