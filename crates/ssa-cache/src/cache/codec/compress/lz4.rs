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
    use rand::{RngCore, SeedableRng};

    use super::*;
    use crate::{
        Result,
        cache::codec::{
            CodecEngine,
            compress::{
                CompressedCodec,
                fixtures::{BytesRaw, assert_raw_header},
            },
        },
    };

    fn assert_lz4_header(encoded: &[u8]) {
        assert!(!encoded.is_empty());
        assert_eq!(encoded[0] & 0b1111_0000, 0b0001_0000);
        assert_eq!(encoded[0] & 0b0000_1111, 0b0001);
    }

    #[test]
    fn works_without_bitcode() -> Result<()> {
        type Lz4BytesEngine = CompressedCodec<BytesRaw, Lz4>;

        let value = vec![0u8; 96 * 1024];
        let mut engine = Lz4BytesEngine::default();
        let encoded = engine.encode(&value).to_vec();
        assert_lz4_header(&encoded);

        let decoded: Vec<u8> = engine.decode(&encoded)?;
        assert_eq!(decoded, value);
        Ok(())
    }

    #[cfg(feature = "bitcode")]
    #[test]
    fn small_value_stays_uncompressed() -> Result<()> {
        use crate::cache::codec::bitcode::Bitcode;
        type Lz4Engine = CompressedCodec<Bitcode, Lz4>;

        let value = vec![7u8; 1024];
        let mut engine = Lz4Engine::default();
        let encoded = engine.encode(&value).to_vec();
        assert_raw_header(&encoded);

        let decoded: Vec<u8> = engine.decode(&encoded)?;
        assert_eq!(decoded, value);
        Ok(())
    }

    #[cfg(feature = "bitcode")]
    #[test]
    fn large_compressible_value_is_compressed() -> Result<()> {
        use crate::cache::codec::bitcode::Bitcode;
        type Lz4Engine = CompressedCodec<Bitcode, Lz4>;

        let value = "a".repeat(96 * 1024);
        let mut engine = Lz4Engine::default();
        let encoded = engine.encode(&value).to_vec();
        assert_lz4_header(&encoded);

        let decoded: String = engine.decode(&encoded)?;
        assert_eq!(decoded, value);
        Ok(())
    }

    #[cfg(feature = "bitcode")]
    #[test]
    fn incompressible_data_falls_back_to_raw() -> Result<()> {
        use crate::cache::codec::bitcode::Bitcode;
        type Lz4Engine = CompressedCodec<Bitcode, Lz4>;

        let mut value = vec![0u8; 96 * 1024];
        rand::rngs::StdRng::seed_from_u64(0x5A17).fill_bytes(&mut value);

        let mut engine = Lz4Engine::default();
        let encoded = engine.encode(&value).to_vec();
        assert_raw_header(&encoded);

        let decoded: Vec<u8> = engine.decode(&encoded)?;
        assert_eq!(decoded, value);
        Ok(())
    }

    #[cfg(feature = "bitcode")]
    #[test]
    fn decode_invalid_header_or_truncated_payload() {
        use crate::cache::codec::bitcode::Bitcode;
        type Lz4Engine = CompressedCodec<Bitcode, Lz4>;

        let mut engine = Lz4Engine::default();

        let wrong_version = vec![0b0000_0000, 0x12];
        assert!(matches!(
            engine.decode(&wrong_version),
            Err::<u8, _>(crate::Error::Compress(_))
        ));

        let unknown_algorithm = vec![0b0001_0010, 0x12, 0x34];
        assert!(matches!(
            engine.decode(&unknown_algorithm),
            Err::<u8, _>(crate::Error::Compress(_))
        ));

        let truncated_lz4_len = vec![0b0001_0001, 0x01, 0x02, 0x03];
        assert!(matches!(
            engine.decode(&truncated_lz4_len),
            Err::<u8, _>(crate::Error::Compress(_))
        ));

        let invalid_lz4_payload = vec![
            0b0001_0001,
            0x08,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0xAA,
            0xBB,
        ];
        assert!(matches!(
            engine.decode(&invalid_lz4_payload),
            Err::<u8, _>(crate::Error::Compress(_))
        ));
    }

    #[cfg(feature = "bitcode")]
    #[test]
    fn same_type_roundtrip_with_multiple_engines() -> Result<()> {
        use crate::cache::codec::bitcode::Bitcode;
        type Lz4Engine = CompressedCodec<Bitcode, Lz4>;

        let value = "abc".repeat(1024);

        let mut bitcode_engine = Bitcode::default();
        let encoded = bitcode_engine.encode(&value).to_vec();
        let decoded: String = bitcode_engine.decode(&encoded)?;
        assert_eq!(decoded, value);

        let mut lz4_engine = Lz4Engine::default();
        let encoded = lz4_engine.encode(&value).to_vec();
        let decoded: String = lz4_engine.decode(&encoded)?;
        assert_eq!(decoded, value);

        Ok(())
    }
}
