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
            CodecEngine, SkipReason,
            compress::{
                CompressFrame, CompressedCodec, Error,
                fixtures::{BytesRaw, SizedBytesRaw, assert_raw_header},
            },
            fixtures::FixtureEngine,
        },
    };

    #[track_caller]
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
        let encoded = engine.encode(&value).unwrap().to_vec();
        assert_lz4_header(&encoded);

        let decoded: Vec<u8> = engine.decode(&encoded)?;
        assert_eq!(decoded, value);
        Ok(())
    }

    #[test]
    fn small_value_stays_uncompressed() -> Result<()> {
        type Lz4Engine = CompressedCodec<FixtureEngine, Lz4>;

        let value = vec![7u8; 1024];
        let mut engine = Lz4Engine::default();
        let encoded = engine.encode(&value).unwrap().to_vec();
        assert_raw_header(&encoded);

        let decoded: Vec<u8> = engine.decode(&encoded)?;
        assert_eq!(decoded, value);
        Ok(())
    }

    #[test]
    fn large_compressible_value_is_compressed() -> Result<()> {
        type Lz4Engine = CompressedCodec<FixtureEngine, Lz4>;

        let value = "a".repeat(96 * 1024);
        let mut engine = Lz4Engine::default();
        let encoded = engine.encode(&value).unwrap().to_vec();
        assert_lz4_header(&encoded);

        let decoded: String = engine.decode(&encoded)?;
        assert_eq!(decoded, value);
        Ok(())
    }

    #[test]
    fn corrupted_compressed_payload_returns_checksum_mismatch() {
        type Lz4Engine = CompressedCodec<FixtureEngine, Lz4>;

        let value = "a".repeat(96 * 1024);
        let mut engine = Lz4Engine::default();
        let mut encoded = engine.encode(&value).unwrap().to_vec();
        let payload_index = CompressFrame::<Lz4>::COMPRESSED_HEADER_LEN + 1;
        encoded[payload_index] ^= 0x01;

        let err: crate::Result<String> = engine.decode(&encoded);
        let err = err.unwrap_err();
        assert!(matches!(
            err,
            crate::Error::Codec(crate::cache::codec::Error::Compress(
                Error::ChecksumMismatch
            ))
        ));
    }

    #[test]
    fn corrupted_declared_length_returns_checksum_mismatch() {
        type Lz4Engine = CompressedCodec<FixtureEngine, Lz4>;

        let value = "a".repeat(96 * 1024);
        let mut engine = Lz4Engine::default();
        let mut encoded = engine.encode(&value).unwrap().to_vec();
        encoded[1] ^= 0x01;

        let err: crate::Result<String> = engine.decode(&encoded);
        let err = err.unwrap_err();
        assert!(matches!(
            err,
            crate::Error::Codec(crate::cache::codec::Error::Compress(
                Error::ChecksumMismatch
            ))
        ));
    }

    #[test]
    fn compressed_header_flip_to_raw_returns_checksum_mismatch() {
        type Lz4Engine = CompressedCodec<FixtureEngine, Lz4>;

        let value = "a".repeat(96 * 1024);
        let mut engine = Lz4Engine::default();
        let mut encoded = engine.encode(&value).unwrap().to_vec();
        encoded[0] = 0b0001_0000;

        let err: crate::Result<String> = engine.decode(&encoded);
        let err = err.unwrap_err();
        assert!(matches!(
            err,
            crate::Error::Codec(crate::cache::codec::Error::Compress(
                Error::ChecksumMismatch
            ))
        ));
    }

    #[test]
    fn incompressible_data_falls_back_to_raw() -> Result<()> {
        type Lz4Engine = CompressedCodec<FixtureEngine, Lz4>;

        let mut value = vec![0u8; 96 * 1024];
        rand::rngs::StdRng::seed_from_u64(0x5A17).fill_bytes(&mut value);

        let mut engine = Lz4Engine::default();
        let encoded = engine.encode(&value).unwrap().to_vec();
        assert_raw_header(&encoded);

        let decoded: Vec<u8> = engine.decode(&encoded)?;
        assert_eq!(decoded, value);
        Ok(())
    }

    #[test]
    fn decode_invalid_header_or_truncated_payload() {
        type Lz4Engine = CompressedCodec<FixtureEngine, Lz4>;

        let mut engine = Lz4Engine::default();

        let wrong_version = vec![0b0000_0000, 0x12];
        assert!(matches!(
            engine.decode(&wrong_version),
            Err::<u8, _>(crate::Error::Codec(crate::cache::codec::Error::Compress(_)))
        ));

        let unknown_algorithm = vec![0b0001_0010, 0x12, 0x34];
        assert!(matches!(
            engine.decode(&unknown_algorithm),
            Err::<u8, _>(crate::Error::Codec(crate::cache::codec::Error::Compress(_)))
        ));

        let truncated_lz4_len = vec![0b0001_0001, 0x01, 0x02, 0x03, 0x04];
        assert!(matches!(
            engine.decode(&truncated_lz4_len),
            Err::<u8, _>(crate::Error::Codec(crate::cache::codec::Error::Compress(_)))
        ));

        let header = 0b0001_0001;
        let len_bytes = (8u32).to_le_bytes();
        let payload = [0xAA, 0xBB];
        let checksum = CompressFrame::<Lz4>::checksum(
            super::super::Header::from_inner(header),
            &len_bytes,
            &payload,
        )
        .to_le_bytes();
        let mut invalid_lz4_payload = vec![header];
        invalid_lz4_payload.extend_from_slice(&len_bytes);
        invalid_lz4_payload.extend_from_slice(&checksum);
        invalid_lz4_payload.extend_from_slice(&payload);
        assert!(matches!(
            engine.decode(&invalid_lz4_payload),
            Err::<u8, _>(crate::Error::Codec(crate::cache::codec::Error::Compress(_)))
        ));
    }

    #[test]
    fn same_type_roundtrip_with_multiple_engines() -> Result<()> {
        type Lz4Engine = CompressedCodec<FixtureEngine, Lz4>;

        let value = "abc".repeat(1024);

        let mut raw_engine = FixtureEngine::default();
        let encoded = raw_engine.encode(&value).unwrap().to_vec();
        let decoded: String = raw_engine.decode(&encoded)?;
        assert_eq!(decoded, value);

        let mut lz4_engine = Lz4Engine::default();
        let encoded = lz4_engine.encode(&value).unwrap().to_vec();
        let decoded: String = lz4_engine.decode(&encoded)?;
        assert_eq!(decoded, value);

        Ok(())
    }

    #[test]
    fn oversize_raw_payload_skips_cache_encoding() {
        type Lz4SizedEngine = CompressedCodec<SizedBytesRaw, Lz4>;

        let oversize_len = CompressFrame::<Lz4>::MAX_DECOMPRESSED_LEN + 1;
        let mut engine = Lz4SizedEngine::default();
        assert!(matches!(
            engine.encode(&oversize_len),
            Err(SkipReason::EncodedValueTooLarge { .. })
        ));
    }
}
