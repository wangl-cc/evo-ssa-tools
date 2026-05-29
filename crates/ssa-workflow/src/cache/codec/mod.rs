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
//! | `Bitcode06` | `bitcode06` | You want a fast binary format tied to `bitcode` 0.6. |
//! | `Postcard` | `postcard` | You want a compact serde format with a published wire spec. |
//! | [`CheckedCodec<E>`] | always | You want CRC32C corruption checks around another codec. |
//! | `CompressedCodec<E, C, P>` | `compress` | You want framed compression around another codec. |
//!
//! # Value format names
//!
//! Format names are part of the persistent cache contract. Use stable, versioned names such as
//! `"bitcode06-v1"`, `"postcard-v1"`, `"checked-v1"`, or `"lz4-v1"`.
//!
//! Use `-` inside one segment and never `--`. The `--` separator is reserved for composed
//! formats built with [`ValueFormat::concat`].
//!
//! Codec wrappers append their own segment to the inner format:
//!
//! ```ignore
//! // Renders as "bitcode06-v1--checked-v1"
//! const VALUE_FORMAT: ValueFormat =
//!     ValueFormat::concat(&Bitcode06::<MyType>::VALUE_FORMAT, "checked-v1");
//! ```
//!
//! Segment constructors assert this convention, so displayed names remain unambiguous.

use crate::identity::{IdentifierSegmentChain, assert_identifier_segment};

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
/// Equality is structural: segment boundaries matter. Segment constructors reject `--`, so each
/// displayed format maps back to one unambiguous segment chain.
#[derive(Clone, Copy, Debug)]
pub struct ValueFormat(ValueFormatRepr);

#[derive(Clone, Copy, Debug)]
enum ValueFormatRepr {
    Root(&'static str),
    Child {
        parent: &'static ValueFormat,
        segment: &'static str,
    },
}

use ValueFormatRepr::{Child, Root};

impl ValueFormat {
    /// Create a value format from one stable segment.
    pub const fn new(name: &'static str) -> Self {
        assert_identifier_segment(name, false);
        Self(Root(name))
    }

    /// Append one stable segment to an existing format.
    ///
    /// The rendered form inserts `--` between segments: `base--suffix`.
    /// Callers should **not** include the separator in `suffix`.
    pub const fn concat(base: &'static ValueFormat, suffix: &'static str) -> Self {
        assert_identifier_segment(suffix, false);
        Self(Child {
            parent: base,
            segment: suffix,
        })
    }
}

impl IdentifierSegmentChain for ValueFormat {
    fn for_each_segment(&self, mut visit: impl FnMut(&str)) {
        fn walk(format: &ValueFormat, visit: &mut impl FnMut(&str)) {
            match format.0 {
                Root(value) => visit(value),
                Child { parent, segment } => {
                    walk(parent, visit);
                    visit(segment);
                }
            }
        }

        walk(self, &mut visit);
    }
}

impl std::fmt::Display for ValueFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.write_segments("--", f)
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
            (Root(a), Root(b)) => a == b,
            (
                Child {
                    parent: a_parent,
                    segment: a_segment,
                },
                Child {
                    parent: b_parent,
                    segment: b_segment,
                },
            ) => a_segment == b_segment && a_parent.0 == b_parent.0,
            _ => false,
        }
    }
}

impl Eq for ValueFormatRepr {}

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
    /// never `--`. For wrappers, compose with [`ValueFormat::concat`]:
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
    fn display_with_double_hyphen_separator() {
        const FORMAT: ValueFormat = ValueFormat::new("bitcode06-v1");
        const CHECKED: ValueFormat = ValueFormat::concat(&FORMAT, "checked-v1");
        const COMPRESSED: ValueFormat = ValueFormat::concat(&CHECKED, "zstd-v1");

        assert_eq!(CHECKED.to_string(), "bitcode06-v1--checked-v1");
        assert_eq!(COMPRESSED.to_string(), "bitcode06-v1--checked-v1--zstd-v1");
    }

    #[test]
    fn concat_with_same_chain_is_equal() {
        const BASE: ValueFormat = ValueFormat::new("bitcode06-v1");
        const A: ValueFormat = ValueFormat::concat(&BASE, "checked-v1");
        const B: ValueFormat = ValueFormat::concat(&BASE, "checked-v1");

        assert_eq!(A, B);
    }

    #[test]
    #[should_panic(expected = "identifier segment must not contain `--`")]
    fn value_format_rejects_double_hyphen_segment() {
        let _ = ValueFormat::new("bitcode06-v1--checked-v1");
    }

    #[test]
    #[should_panic(expected = "identifier segment contains an invalid character")]
    fn value_format_rejects_slash_segment() {
        let _ = ValueFormat::new("bitcode06-v1/checked-v1");
    }

    #[test]
    #[should_panic(expected = "identifier segment must not contain `--`")]
    fn value_format_concat_rejects_double_hyphen_suffix() {
        const BASE: ValueFormat = ValueFormat::new("bitcode06-v1");
        let _ = ValueFormat::concat(&BASE, "checked--v1");
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
