//! Data-block encoding and decoding.
//!
//! A block is the physical read and checksum unit inside a segment. Block
//! length and record count live in the segment block index; block-local
//! decoding metadata lives in a fixed-size footer at the end of the block,
//! with a CRC32C over all previous block bytes as the final field.
//!
//! The module splits along the data flow:
//!
//! - [`layout`]: the byte-layout math shared by both directions (block
//!   footer, value region, value index)
//! - [`encode`]: sorted entries in, block bytes out
//! - [`decode`]: block bytes in, zero-copy records out

mod decode;
mod encode;
mod layout;

pub(crate) use decode::{DecodedBlock, ParsedRecord};
pub(crate) use encode::BlockBuilder;
pub(crate) use layout::BLOCK_FOOTER_LEN;
