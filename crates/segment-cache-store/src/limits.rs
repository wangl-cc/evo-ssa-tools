//! Implementation allocation budgets for the v1 physical format.

pub(crate) const MAX_KEY_LEN: usize = 64 * 1024;
pub(crate) const MAX_VALUE_LEN: usize = 64 * 1024 * 1024;
pub(crate) const MAX_VALUE_PAYLOAD_LEN: usize = 64 * 1024 * 1024;
pub(crate) const MAX_ENCODED_BLOCK_LEN: usize = 128 * 1024 * 1024;
pub(crate) const MAX_FOOTER_LEN: usize = 64 * 1024 * 1024;
pub(crate) const MAX_MANIFEST_LEN: usize = 256 * 1024 * 1024;
pub(crate) const MAX_BLOCK_COUNT: usize = 1_048_576;
