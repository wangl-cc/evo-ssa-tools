//! Narrow key-byte algorithms shared by block and segment encoding.

/// Length of the byte prefix shared by `left` and `right`.
pub(crate) fn common_prefix_len(left: &[u8], right: &[u8]) -> usize {
    left.iter()
        .zip(right)
        .take_while(|(left, right)| left == right)
        .count()
}
