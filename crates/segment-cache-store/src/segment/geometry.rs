//! Resolved physical geometry shared by every segment in one store.

use crate::{
    block::{BlockChecksumKind, ValuePayloadCompressionKind},
    value::ValueLayout,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SegmentGeometry {
    pub(crate) key_len: usize,
    pub(crate) value_layout: ValueLayout,
    pub(crate) block_checksum: BlockChecksumKind,
    pub(crate) value_payload_compression: ValuePayloadCompressionKind,
}
