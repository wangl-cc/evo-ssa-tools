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
    /// - `2`: [`Zstd`] when the `zstd` feature is enabled
    /// - `3..=15`: currently unassigned
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
    /// # Safety
    ///
    /// `output` must be at least `Self::max_output_size(input.len())` bytes long.
    /// [`super::CompressedCodec`] upholds this invariant before every call.
    unsafe fn compress_into_unchecked(&mut self, input: &[u8], output: &mut [u8]) -> usize;

    /// Decompress `input` into `output` and return the decompressed length.
    ///
    /// # Safety
    ///
    /// `output` must be large enough to hold the full decompressed data for `input`.
    /// [`super::CompressedCodec`] upholds this invariant before every call.
    unsafe fn decompress_into_unchecked(
        &mut self,
        input: &[u8],
        output: &mut [u8],
    ) -> Result<usize, Error>;
}

#[cfg(feature = "lz4")]
mod lz4 {
    use super::*;

    #[derive(Debug, Clone, Copy, Default)]
    pub struct Lz4;

    impl crate::cache::Fork for Lz4 {
        fn fork(&self) -> Self {
            *self
        }
    }

    #[cfg(feature = "lz4")]
    impl Compress for Lz4 {
        const ALGORITHM_ID: u8 = 1;

        fn max_output_size(&self, input_len: usize) -> usize {
            lz4_flex::block::get_maximum_output_size(input_len)
        }

        unsafe fn compress_into_unchecked(&mut self, input: &[u8], output: &mut [u8]) -> usize {
            lz4_flex::block::compress_into(input, output)
                .expect("lz4 compress_into should succeed for a max_output_size-sized output")
        }

        unsafe fn decompress_into_unchecked(
            &mut self,
            input: &[u8],
            output: &mut [u8],
        ) -> Result<usize, Error> {
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
            let mut compressor = Lz4;
            let input = vec![7u8; 8 * 1024];
            let mut compressed = vec![0u8; compressor.max_output_size(input.len())];
            let compressed_len =
                unsafe { compressor.compress_into_unchecked(&input, &mut compressed) };

            let mut decompressed = vec![0u8; input.len()];
            let decompressed_len = unsafe {
                compressor
                    .decompress_into_unchecked(&compressed[..compressed_len], &mut decompressed)?
            };

            assert_eq!(decompressed_len, input.len());
            assert_eq!(decompressed, input);
            Ok(())
        }

        #[test]
        fn lz4_invalid_block_returns_error() {
            let mut compressor = Lz4;
            let input = vec![5u8; 256];
            let mut compressed = vec![0u8; compressor.max_output_size(input.len())];
            let compressed_len =
                unsafe { compressor.compress_into_unchecked(&input, &mut compressed) };
            let mut output = vec![0u8; input.len() - 1];
            assert!(matches!(
                unsafe {
                    compressor.decompress_into_unchecked(&compressed[..compressed_len], &mut output)
                },
                Err(Error::Lz4(_))
            ));
        }

        #[test]
        fn lz4_fork_produces_working_compressor() -> Result<(), CodecError> {
            use crate::cache::Fork;
            let original = Lz4;
            let mut forked = original.fork();
            let input = vec![7u8; 4 * 1024];
            let mut compressed = vec![0u8; forked.max_output_size(input.len())];
            let compressed_len =
                unsafe { forked.compress_into_unchecked(&input, &mut compressed) };
            let mut decompressed = vec![0u8; input.len()];
            let decompressed_len = unsafe {
                forked
                    .decompress_into_unchecked(&compressed[..compressed_len], &mut decompressed)?
            };
            assert_eq!(decompressed_len, input.len());
            assert_eq!(decompressed, input);
            Ok(())
        }
    }
}
#[cfg(feature = "lz4")]
pub use lz4::Lz4;

#[cfg(feature = "zstd")]
mod zstd_support {
    use std::io;

    use super::*;

    pub struct Zstd {
        level: i32,
        compressor: zstd::bulk::Compressor<'static>,
        decompressor: zstd::bulk::Decompressor<'static>,
    }

    impl std::fmt::Debug for Zstd {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("Zstd").field("level", &self.level).finish()
        }
    }

    impl Zstd {
        pub fn new(level: i32) -> io::Result<Self> {
            Ok(Self {
                level,
                compressor: zstd::bulk::Compressor::new(level)?,
                decompressor: zstd::bulk::Decompressor::new()?,
            })
        }
    }

    impl crate::cache::Fork for Zstd {
        fn fork(&self) -> Self {
            Self::new(self.level).expect("zstd context creation should succeed for a previously valid level")
        }
    }

    impl Compress for Zstd {
        const ALGORITHM_ID: u8 = 2;

        fn max_output_size(&self, input_len: usize) -> usize {
            zstd::zstd_safe::compress_bound(input_len)
        }

        unsafe fn compress_into_unchecked(&mut self, input: &[u8], output: &mut [u8]) -> usize {
            self.compressor
                .compress_to_buffer(input, output)
                .expect("zstd compress_to_buffer should succeed for a max_output_size-sized output")
        }

        unsafe fn decompress_into_unchecked(
            &mut self,
            input: &[u8],
            output: &mut [u8],
        ) -> Result<usize, Error> {
            Ok(self.decompressor.decompress_to_buffer(input, output)?)
        }
    }

    #[cfg(test)]
    #[cfg_attr(coverage_nightly, coverage(off))]
    mod tests {
        use super::*;
        use crate::cache::codec::Error as CodecError;

        #[test]
        fn zstd_block_roundtrip() -> Result<(), CodecError> {
            let mut compressor = Zstd::new(3).expect("zstd config should be valid");
            let input = vec![7u8; 8 * 1024];
            let mut compressed = vec![0u8; compressor.max_output_size(input.len())];
            let compressed_len =
                unsafe { compressor.compress_into_unchecked(&input, &mut compressed) };

            let mut decompressed = vec![0u8; input.len()];
            let decompressed_len = unsafe {
                compressor
                    .decompress_into_unchecked(&compressed[..compressed_len], &mut decompressed)?
            };

            assert_eq!(decompressed_len, input.len());
            assert_eq!(decompressed, input);
            Ok(())
        }

        #[test]
        fn zstd_roundtrip_uses_explicit_level() -> Result<(), CodecError> {
            let mut compressor = Zstd::new(3).expect("zstd config should be valid");
            let input = vec![9u8; 4 * 1024];
            let mut compressed = vec![0u8; compressor.max_output_size(input.len())];
            let compressed_len =
                unsafe { compressor.compress_into_unchecked(&input, &mut compressed) };
            let mut decompressed = vec![0u8; input.len()];
            let decompressed_len = unsafe {
                compressor
                    .decompress_into_unchecked(&compressed[..compressed_len], &mut decompressed)?
            };

            assert_eq!(decompressed_len, input.len());
            assert_eq!(decompressed, input);
            Ok(())
        }

        #[test]
        fn zstd_invalid_block_returns_error() {
            let mut compressor = Zstd::new(3).expect("zstd config should be valid");
            let mut output = vec![0u8; 32];
            assert!(matches!(
                unsafe { compressor.decompress_into_unchecked(b"not-zstd", &mut output) },
                Err(Error::Zstd(_))
            ));
        }

        #[test]
        fn zstd_level_only_configuration_roundtrip() -> Result<(), CodecError> {
            let mut compressor = Zstd::new(5).expect("zstd config should be valid");
            let input = vec![11u8; 1024];
            let mut compressed = vec![0u8; compressor.max_output_size(input.len())];
            let compressed_len =
                unsafe { compressor.compress_into_unchecked(&input, &mut compressed) };
            let mut decompressed = vec![0u8; input.len()];
            let decompressed_len = unsafe {
                compressor
                    .decompress_into_unchecked(&compressed[..compressed_len], &mut decompressed)?
            };

            assert_eq!(decompressed_len, input.len());
            assert_eq!(decompressed, input);
            Ok(())
        }

        #[test]
        fn zstd_debug_reports_config() {
            let compressor = Zstd::new(6).expect("zstd config should be valid");

            let debug = format!("{compressor:?}");
            assert!(debug.contains("Zstd"));
        }

        #[test]
        fn zstd_fork_produces_working_compressor() -> Result<(), CodecError> {
            use crate::cache::Fork;
            let original = Zstd::new(3).expect("zstd config should be valid");
            let mut forked = original.fork();
            let input = vec![7u8; 4 * 1024];
            let mut compressed = vec![0u8; forked.max_output_size(input.len())];
            let compressed_len =
                unsafe { forked.compress_into_unchecked(&input, &mut compressed) };
            let mut decompressed = vec![0u8; input.len()];
            let decompressed_len = unsafe {
                forked
                    .decompress_into_unchecked(&compressed[..compressed_len], &mut decompressed)?
            };
            assert_eq!(decompressed_len, input.len());
            assert_eq!(decompressed, input);
            Ok(())
        }
    }
}

#[cfg(feature = "zstd")]
pub use zstd_support::Zstd;
