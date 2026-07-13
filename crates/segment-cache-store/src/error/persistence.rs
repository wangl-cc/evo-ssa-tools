//! Persistent representation error vocabulary.
//!
//! These errors describe block and segment codec boundaries: decoding fails as
//! corruption and encoding fails when data exceeds persisted format limits.
//! Aggregation into the public [`crate::Error`] happens in the parent module.

/// Encoding would exceed the v1 on-disk format capacity.
///
/// Every overflow is the same caller decision — the data being written does
/// not fit the format's `u32`-based envelope — so this is one type carrying
/// the overflowing quantity as a diagnostic, not a variant per field.
#[derive(thiserror::Error, Clone, Debug, Eq, PartialEq)]
#[error("{quantity} exceeds the v1 on-disk format limit")]
pub struct FormatError {
    quantity: &'static str,
}

impl FormatError {
    pub(crate) fn limit(quantity: &'static str) -> Self {
        Self { quantity }
    }

    /// The encoded quantity that overflowed, e.g. `"block length"`.
    #[must_use]
    pub fn quantity(&self) -> &'static str {
        self.quantity
    }
}

/// Published cache data violates a physical or logical storage invariant.
///
/// Read paths may degrade unreadable block or segment bytes to a cache miss.
/// Logical snapshot violations are surfaced because choosing one value would
/// invent data semantics.
#[derive(thiserror::Error, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum CorruptionError {
    #[error("corrupt or malformed block")]
    Block,

    #[error("corrupt, malformed, or unsupported segment format")]
    SegmentFormat,

    #[error("duplicate visible key violates the store's unique-key invariant")]
    DuplicateVisibleKey,
}

/// Failure while streaming one segment into a sink: either the sink failed or
/// the data exceeds format limits.
///
/// Crate-internal: the public API surfaces both cases through [`crate::Error`].
#[derive(thiserror::Error, Debug)]
pub(crate) enum SegmentWriteError {
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Format(#[from] FormatError),
}
