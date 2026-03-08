use super::Compress;

#[derive(Debug, Clone, Copy, Default)]
pub struct Lz4;

impl Compress for Lz4 {
    const ALGORITHM_ID: u8 = 1;

    fn max_output_size(input_len: usize) -> usize {
        lz4_flex::block::get_maximum_output_size(input_len)
    }

    fn compress_into(input: &[u8], output: &mut [u8]) -> usize {
        lz4_flex::block::compress_into(input, output)
            .expect("lz4 compress_into output buffer too small for max_output_size contract")
    }

    fn decompress_into(input: &[u8], output: &mut [u8]) -> Result<usize, super::Error> {
        Ok(lz4_flex::block::decompress_into(input, output)?)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::Result;

    #[test]
    fn block_roundtrip() -> Result<()> {
        let input = vec![7u8; 8 * 1024];
        let mut compressed = vec![0u8; Lz4::max_output_size(input.len())];
        let compressed_len = Lz4::compress_into(&input, &mut compressed);

        let mut decompressed = vec![0u8; input.len()];
        let decompressed_len =
            Lz4::decompress_into(&compressed[..compressed_len], &mut decompressed)?;

        assert_eq!(decompressed_len, input.len());
        assert_eq!(decompressed, input);
        Ok(())
    }
}
