#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
#![cfg_attr(
    coverage_nightly,
    allow(
        unused_features,
        reason = "coverage_nightly enables coverage_attribute for coverage(off) test annotations"
    )
)]

//! Stable schema fingerprints for cache wire formats.
//!
//! `CacheSchema` describes the serialized shape of a type, not its Rust memory layout. The
//! resulting canonical bytes are hashed with BLAKE3 and truncated to 128 bits.
//!
//! ```rust
//! # #[cfg(feature = "derive")]
//! use ssa_cache_schema::{CacheSchema, schema_fingerprint};
//!
//! # #[cfg(feature = "derive")]
//! #[derive(CacheSchema)]
//! struct Params {
//!     width: u32,
//!     height: u32,
//! }
//!
//! # #[cfg(feature = "derive")]
//! let first = schema_fingerprint::<Params>();
//! # #[cfg(feature = "derive")]
//! let second = schema_fingerprint::<Params>();
//! # #[cfg(feature = "derive")]
//! assert_eq!(first, second);
//! ```
//!
//! Serde attributes are intentionally ignored by the derive macro; use `cache_schema`
//! attributes or a manual implementation when serde behavior should affect the cache schema:
//!
//! ```rust
//! # #[cfg(feature = "derive")]
//! # mod example {
//! # #[cfg(feature = "derive")]
//! use ssa_cache_schema::CacheSchema;
//!
//! #[derive(CacheSchema)]
//! struct WithSerdeAttr {
//!     #[serde(skip)]
//!     value: u32,
//! }
//! # }
//! ```
//!
//! Unsupported `cache_schema` attributes are rejected instead of being ignored:
//!
//! ```compile_fail
//! use ssa_cache_schema::CacheSchema;
//!
//! #[derive(CacheSchema)]
//! struct Bad {
//!     #[cache_schema(skip)]
//!     value: u32,
//! }
//! ```
//!
//! Recursive schemas are not expanded automatically. Derive does not try to reject every recursive
//! type shape, so recursive cache schemas need a manual implementation or an explicit reference
//! scheme.

mod impls;
mod writer;

#[cfg(feature = "derive")]
pub use ssa_cache_schema_derive::CacheSchema;
pub use writer::{EmptyProductStyle, SchemaWriter};

/// A 128-bit schema fingerprint.
pub type SchemaFingerprint = [u8; 16];

/// A type that can describe its cache wire schema.
pub trait CacheSchema {
    /// Write this type's canonical schema description.
    fn write_schema(w: &mut SchemaWriter);
}

/// Compute the BLAKE3-128 schema fingerprint for `T`.
pub fn schema_fingerprint<T: CacheSchema + ?Sized>() -> SchemaFingerprint {
    let mut writer = SchemaWriter::new();
    T::write_schema(&mut writer);
    writer.finish_fingerprint()
}
