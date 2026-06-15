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
//! - [`decode`]: block bytes in, zero-copy records out

mod decode;
mod encode;
mod layout;

pub(crate) use decode::{DecodedBlock, ParsedRecord};
pub(super) use encode::BlockBuilder;
pub(crate) use layout::{BlockLookupLayout, KEY_PREFIX_LEN_LEN};
