//! Writer policy for value-payload compression.

/// Writer-side policy for deciding whether a value payload is worth compressing.
///
/// This policy is not persisted. It only controls newly written blocks. The
/// persisted store compression kind still determines which frame encodings a
/// reader must support, and each block records whether its own payload was
/// stored raw or compressed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ValuePayloadCompressionPolicy {
    /// Minimum raw value-payload bytes before compression is attempted.
    min_try_len: usize,
    /// Minimum saved percentage required to keep a compressed frame.
    min_saved_percent: u8,
}

/// Invalid writer-side value-payload compression policy.
#[derive(thiserror::Error, Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum CompressionPolicyError {
    #[error("min_saved_percent must be at most 100")]
    MinSavedPercentTooLarge,
}

impl Default for ValuePayloadCompressionPolicy {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl ValuePayloadCompressionPolicy {
    /// Default writer-side compression policy used by [`crate::CommitOptions`].
    pub const DEFAULT: Self = Self {
        min_try_len: 64 * 1024,
        min_saved_percent: 20,
    };

    /// Creates a policy from explicit thresholds.
    ///
    /// # Errors
    ///
    /// Returns [`CompressionPolicyError::MinSavedPercentTooLarge`] when
    /// `min_saved_percent` is greater than 100.
    pub const fn new(
        min_try_len: usize,
        min_saved_percent: u8,
    ) -> Result<Self, CompressionPolicyError> {
        if min_saved_percent > 100 {
            return Err(CompressionPolicyError::MinSavedPercentTooLarge);
        }
        Ok(Self {
            min_try_len,
            min_saved_percent,
        })
    }

    /// Sets the minimum raw value-payload bytes before compression is attempted.
    pub const fn with_min_try_len(mut self, min_try_len: usize) -> Self {
        self.min_try_len = min_try_len;
        self
    }

    /// Sets the minimum saved percentage required to keep a compressed frame.
    ///
    /// # Errors
    ///
    /// Returns [`CompressionPolicyError::MinSavedPercentTooLarge`] when
    /// `min_saved_percent` is greater than 100.
    pub const fn with_min_saved_percent(
        self,
        min_saved_percent: u8,
    ) -> Result<Self, CompressionPolicyError> {
        Self::new(self.min_try_len, min_saved_percent)
    }

    /// Returns the minimum raw value-payload bytes before compression is attempted.
    #[must_use]
    pub const fn min_try_len(self) -> usize {
        self.min_try_len
    }

    /// Returns the minimum saved percentage required to keep a compressed frame.
    #[must_use]
    pub const fn min_saved_percent(self) -> u8 {
        self.min_saved_percent
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_rejects_invalid_saved_percentage() {
        assert_eq!(
            ValuePayloadCompressionPolicy::new(0, 101),
            Err(CompressionPolicyError::MinSavedPercentTooLarge)
        );
    }
}
