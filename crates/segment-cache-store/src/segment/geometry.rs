//! Resolved physical geometry shared by every segment in one store.
//!
//! Geometry is split into two levels:
//!
//! - [`StoreSchema`]: the logical data contract (key width, value layout). Two stores must share
//!   the same schema for any inter-store operation.
//! - [`SegmentEncoding`]: the physical value-payload compression choice. This may differ between
//!   stores; a merge across different encodings transcodes rather than copying files.

use crate::{block::ValuePayloadCompressionKind, value::ValueLayout};

/// Logical data contract shared by every record in one store.
///
/// Two stores must have the same schema to be merge-compatible.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct StoreSchema {
    pub(crate) key_len: usize,
    pub(crate) value_layout: ValueLayout,
}

/// Physical byte-level encoding choices for segment files.
///
/// These may differ between stores without affecting logical compatibility.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SegmentEncoding {
    pub(crate) value_payload_compression: ValuePayloadCompressionKind,
}

/// Complete physical geometry of one segment file.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SegmentGeometry {
    pub(crate) schema: StoreSchema,
    pub(crate) encoding: SegmentEncoding,
}
