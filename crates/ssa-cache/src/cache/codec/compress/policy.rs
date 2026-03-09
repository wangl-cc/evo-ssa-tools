/// Compression action selected by [`CompressPolicy`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionAction {
    /// Keep the raw frame.
    Raw,
    /// Keep or attempt the compressed frame.
    Compress,
}

/// Policy hooks that control pre-compression routing and post-compression acceptance.
pub trait CompressPolicy {
    /// Decide what to do before compression based on the serialized payload size.
    fn before_compress(&self, raw_size: usize) -> CompressionAction;

    /// Decide whether to keep the compressed frame after compression.
    ///
    /// `raw_frame_size` and `compressed_frame_size` include framing overhead.
    /// Returning [`CompressionAction::Raw`] falls back to the raw frame.
    fn after_compress(
        &self,
        raw_frame_size: usize,
        compressed_frame_size: usize,
    ) -> CompressionAction;
}

/// Default compression policy used by [`super::CompressedCodec`].
///
/// The default heuristic is deliberately simple:
///
/// - payloads smaller than `min_try_compress_len` stay raw
/// - larger payloads attempt compression
/// - the compressed frame is kept only if it saves at least `min_saved_ratio` relative to the raw
///   frame size, including framing overhead
///
/// The default values are tuned for cache payloads where compression should be
/// beneficial, not merely possible:
///
/// - `min_try_compress_len = 64 KiB`
/// - `min_saved_ratio = 1 / 10` (at least 10% smaller)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DefaultCompressPolicy {
    min_try_compress_len: usize,
    min_saved_ratio_numerator: usize,
    min_saved_ratio_denominator: usize,
}

impl DefaultCompressPolicy {
    /// Create a policy with the crate defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the smallest serialized payload that should attempt compression.
    pub fn with_min_try_compress_len(mut self, min_try_compress_len: usize) -> Self {
        self.min_try_compress_len = min_try_compress_len;
        self
    }

    /// Set the minimum saved ratio required to keep the compressed frame.
    ///
    /// For example, `with_min_saved_ratio(1, 10)` requires the compressed frame
    /// to save at least 10% relative to the raw frame size.
    pub fn with_min_saved_ratio(mut self, numerator: usize, denominator: usize) -> Self {
        assert!(
            denominator > 0,
            "min saved ratio denominator must be non-zero"
        );
        self.min_saved_ratio_numerator = numerator;
        self.min_saved_ratio_denominator = denominator;
        self
    }
}

impl Default for DefaultCompressPolicy {
    fn default() -> Self {
        Self {
            min_try_compress_len: 64 * 1024,
            min_saved_ratio_numerator: 1,
            min_saved_ratio_denominator: 10,
        }
    }
}

impl CompressPolicy for DefaultCompressPolicy {
    fn before_compress(&self, raw_size: usize) -> CompressionAction {
        if raw_size < self.min_try_compress_len {
            CompressionAction::Raw
        } else {
            CompressionAction::Compress
        }
    }

    fn after_compress(
        &self,
        raw_frame_size: usize,
        compressed_frame_size: usize,
    ) -> CompressionAction {
        let saved_bytes = raw_frame_size.saturating_sub(compressed_frame_size);
        let meets_ratio_threshold = saved_bytes.saturating_mul(self.min_saved_ratio_denominator)
            >= raw_frame_size.saturating_mul(self.min_saved_ratio_numerator);

        if !meets_ratio_threshold {
            CompressionAction::Raw
        } else {
            CompressionAction::Compress
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn default_policy_keeps_small_payloads_raw() {
        let decision = DefaultCompressPolicy::default().before_compress(1024);
        assert_eq!(decision, CompressionAction::Raw);
    }

    #[test]
    fn default_policy_tries_compress_for_large_payloads() {
        let decision = DefaultCompressPolicy::default().before_compress(96 * 1024);
        assert_eq!(decision, CompressionAction::Compress);
    }

    #[test]
    fn new_policy_matches_default() {
        assert_eq!(
            DefaultCompressPolicy::new(),
            DefaultCompressPolicy::default()
        );
    }

    #[test]
    fn min_try_compress_len_builder_changes_precompression_decision() {
        let policy = DefaultCompressPolicy::new().with_min_try_compress_len(8 * 1024);
        assert_eq!(policy.before_compress(4 * 1024), CompressionAction::Raw);
        assert_eq!(
            policy.before_compress(16 * 1024),
            CompressionAction::Compress
        );
    }

    #[test]
    fn default_policy_requires_meaningful_savings() {
        let policy = DefaultCompressPolicy::default();
        assert_eq!(
            policy.after_compress(10 * 1024, 9_600),
            CompressionAction::Raw
        );
        assert_eq!(
            policy.after_compress(10 * 1024, 8_000),
            CompressionAction::Compress
        );
    }

    #[test]
    fn min_saved_ratio_builder_changes_postcompression_decision() {
        let policy = DefaultCompressPolicy::new().with_min_saved_ratio(1, 4);
        assert_eq!(
            policy.after_compress(10 * 1024, 8_000),
            CompressionAction::Raw
        );
        assert_eq!(
            policy.after_compress(10 * 1024, 7_500),
            CompressionAction::Compress
        );
    }

    #[test]
    #[should_panic(expected = "min saved ratio denominator must be non-zero")]
    fn min_saved_ratio_builder_rejects_zero_denominator() {
        let _ = DefaultCompressPolicy::new().with_min_saved_ratio(1, 0);
    }
}
