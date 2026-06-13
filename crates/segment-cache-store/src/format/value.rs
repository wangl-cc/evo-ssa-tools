//! Store-wide physical value layout.
//!
//! Values are opaque bytes to the store. This module only describes the
//! store-level layout choice persisted in catalog and segment headers:
//! variable-length values or fixed-length values.

use std::num::NonZeroU32;

/// Store-wide physical value layout.
///
/// The representation mirrors the persisted `value_len` field: `0` means
/// variable-length values, and any non-zero `u32` means fixed-length values.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ValueLayout {
    fixed_len: Option<NonZeroU32>,
}

impl ValueLayout {
    /// Values are opaque byte slices with per-record lengths.
    pub const VARIABLE: Self = Self { fixed_len: None };

    /// Creates a fixed-value layout from a non-zero persisted byte length.
    pub fn fixed(value_len: NonZeroU32) -> Self {
        Self {
            fixed_len: Some(value_len),
        }
    }

    pub(super) fn is_variable(self) -> bool {
        self.fixed_len.is_none()
    }

    pub(crate) fn fixed_len(self) -> Option<NonZeroU32> {
        self.fixed_len
    }

    /// Encodes this layout into the persisted `value_len` integer.
    pub(super) fn to_u32(self) -> u32 {
        self.fixed_len.map_or(0, NonZeroU32::get)
    }

    /// Decodes the persisted `value_len` integer.
    pub(super) fn from_u32(value_len: u32) -> Self {
        Self {
            fixed_len: NonZeroU32::new(value_len),
        }
    }

    pub(super) fn fixed_width(self) -> Option<usize> {
        self.fixed_len.map(|len| len.get() as usize)
    }

    pub(super) fn offset_count(self, record_count: usize) -> Option<usize> {
        if self.is_variable() {
            return record_count.checked_add(1);
        }
        Some(0)
    }

    pub(super) fn value_payload_offset(
        self,
        record_count: usize,
        value_region_offset: usize,
    ) -> Option<usize> {
        if self.is_variable() {
            return value_region_offset
                .checked_add(self.offset_count(record_count)?.checked_mul(4)?);
        }
        Some(value_region_offset)
    }
}

impl Default for ValueLayout {
    fn default() -> Self {
        Self::VARIABLE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod persisted_value_len {
        use super::*;

        #[test]
        fn round_trips_layouts() {
            assert_eq!(
                ValueLayout::from_u32(ValueLayout::VARIABLE.to_u32()),
                ValueLayout::VARIABLE
            );
            let fixed =
                ValueLayout::fixed(NonZeroU32::new(32).expect("fixed value length is non-zero"));
            assert_eq!(ValueLayout::from_u32(fixed.to_u32()), fixed);
        }
    }

    mod region_offsets {
        use super::*;

        #[test]
        fn variable_layout_offsets_past_sentinel_index() {
            assert_eq!(ValueLayout::VARIABLE.offset_count(2), Some(3));
            assert_eq!(ValueLayout::VARIABLE.value_payload_offset(2, 16), Some(28));
        }

        #[test]
        fn fixed_layout_payload_starts_at_value_region() {
            let value_layout =
                ValueLayout::fixed(NonZeroU32::new(4).expect("fixed len is non-zero"));
            assert_eq!(value_layout.offset_count(2), Some(0));
            assert_eq!(value_layout.value_payload_offset(2, 16), Some(16));
        }
    }
}
