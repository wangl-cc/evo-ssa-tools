use super::frame::Error;
use crate::Result;

/// Compression algorithm adapter.
///
/// Implementors provide a block-level compress/decompress pair used by [`super::CompressedCodec`].
///
/// This trait does not decide *when* compression should be attempted or skipped.
/// That policy lives in [`super::CompressedCodec`]. `Compress` only describes how to
/// transform one raw byte slice into one compressed byte slice and back.
pub trait Compress {
    /// Algorithm identifier stored in the low 4 bits of the frame header.
    ///
    /// Assigned ids in the current format:
    ///
    /// - `0`: raw/uncompressed payload (reserved by the frame format)
    /// - `1`: [`Lz4`] when the `lz4` feature is enabled
    /// - `2..=15`: currently unassigned
    const ALGORITHM_ID: u8;

    /// Return the maximum output size to compress `input_len` bytes.
    ///
    /// This is used to pre-allocate output buffers before compression.
    ///
    /// # Buffer contract
    ///
    /// This method returns the maximum output size, which is used to pre-allocate buffers.
    /// The actual compressed size may be smaller.
    fn max_output_size(&self, input_len: usize) -> usize;

    /// Compress `input` into `output` and return the compressed length.
    ///
    /// # Buffer contract
    ///
    /// The buffer `output` **must** be at least `Self::max_output_size(input.len())` length.
    /// [`super::CompressedCodec`] upholds this invariant before every call;
    /// implementations may panic if it is violated.
    fn compress_into(&self, input: &[u8], output: &mut [u8]) -> usize;

    /// Decompress `input` into `output` and return the decompressed length.
    ///
    /// # Buffer contract
    ///
    /// The buffer `output` **must** be large enough to hold the decompressed data.
    /// [`super::CompressedCodec`] upholds this invariant before every call;
    /// implementations may panic if it is violated.
    fn decompress_into(&self, input: &[u8], output: &mut [u8]) -> Result<usize, Error>;
}

#[cfg(feature = "lz4")]
mod lz4 {
    use super::*;

    #[derive(Debug, Clone, Copy, Default)]
    pub struct Lz4;

    #[cfg(feature = "lz4")]
    impl Compress for Lz4 {
        const ALGORITHM_ID: u8 = 1;

        fn max_output_size(&self, input_len: usize) -> usize {
            lz4_flex::block::get_maximum_output_size(input_len)
        }

        fn compress_into(&self, input: &[u8], output: &mut [u8]) -> usize {
            lz4_flex::block::compress_into(input, output)
                .expect("lz4 compress_into output buffer too small for max_output_size contract")
        }

        fn decompress_into(&self, input: &[u8], output: &mut [u8]) -> Result<usize, Error> {
            Ok(lz4_flex::block::decompress_into(input, output)?)
        }
    }

    #[cfg(test)]
    #[cfg_attr(coverage_nightly, coverage(off))]
    mod tests {
        use super::*;
        use crate::cache::codec::Error as CodecError;

        #[test]
        fn lz4_block_roundtrip() -> Result<(), CodecError> {
            let compressor = Lz4;
            let input = vec![7u8; 8 * 1024];
            let mut compressed = vec![0u8; compressor.max_output_size(input.len())];
            let compressed_len = compressor.compress_into(&input, &mut compressed);

            let mut decompressed = vec![0u8; input.len()];
            let decompressed_len =
                compressor.decompress_into(&compressed[..compressed_len], &mut decompressed)?;

            assert_eq!(decompressed_len, input.len());
            assert_eq!(decompressed, input);
            Ok(())
        }
    }
}
#[cfg(feature = "lz4")]
pub use lz4::Lz4;
