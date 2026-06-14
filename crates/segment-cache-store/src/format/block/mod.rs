//! Data-block encoding and decoding.
//!
//! A block is the physical read unit inside a segment. Block length and record
//! count live in the segment block index; block-local decoding metadata lives
//! in a fixed-size footer at the end of the block. The footer stores separate
//! CRC32C values for lookup metadata and value payload bytes, so sparse ordered
//! lookup can validate keys and value offsets before deciding whether it needs
//! to read the value payload.
//!
//! The module splits along the data flow:
//!
//! - [`layout`]: the byte-layout math shared by both directions (block footer, value region, value
//!   index)
//! - [`encode`]: sorted entries in, block bytes out
//! - [`decode`]: block bytes in, zero-copy records out

mod decode;
mod encode;
mod layout;

pub(crate) use decode::{DecodedBlock, ParsedRecord};
pub(super) use encode::BlockBuilder;
pub(crate) use layout::{BLOCK_FOOTER_LEN, BlockFooter};
