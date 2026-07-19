//! Data-block encoding and decoding.
//!
//! A block is the physical read unit inside a segment. Block length and record
//! count live in the segment block index. Block-local metadata is section-local:
//! the key section starts with its prefix descriptor, variable-value blocks keep
//! their value offset table beside the key section, and checksums follow the
//! metadata and value payload they protect.
//!
//! The module splits along the data flow:
//!
//! - [`layout`]: byte-layout math shared by both directions
//! - [`encode`]: sorted entries in, block bytes out
//! - [`decode`]: block bytes in, borrowed record views out
//! - [`checksum`]: persisted checksum identity and optional digest calculation
//! - [`compression`]: persisted compression identity and optional policy, frame, and codec logic

mod checksum;
mod compression;
mod decode;
mod encode;
mod layout;

pub use checksum::BlockChecksumKind;
pub use compression::ValuePayloadCompressionKind;
#[cfg(feature = "value-compression")]
pub use compression::{CompressionPolicyError, ValuePayloadCompressionPolicy};
#[cfg(feature = "value-compression")]
pub(crate) use compression::{ValuePayloadDecoder, ValuePayloadEncoder};
pub(crate) use decode::{BlockDecodeOptions, BlockKeyRangeRef, DecodedBlock, ParsedRecord};
pub(super) use encode::BlockBuilder;
pub(crate) use layout::BLOCK_METADATA_HEADER_LEN;
