//! Block-local value-payload storage frame.

#[cfg(any(feature = "value-compression-lz4", feature = "value-compression-zstd"))]
use crate::{
    block::ValuePayloadCompressionPolicy,
    error::{CorruptionError, FormatError},
};

#[cfg(any(feature = "value-compression-lz4", feature = "value-compression-zstd"))]
pub(super) const HEADER_LEN: usize = 1;
#[cfg(any(feature = "value-compression-lz4", feature = "value-compression-zstd"))]
pub(super) const MAX_DECODED_PAYLOAD_LEN: usize = 64 * 1024 * 1024;

#[cfg(any(feature = "value-compression-lz4", feature = "value-compression-zstd"))]
const STORAGE_RAW: u8 = 0;
#[cfg(any(feature = "value-compression-lz4", feature = "value-compression-zstd"))]
const STORAGE_COMPRESSED: u8 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PayloadFrame {
    storage: PayloadStorage,
    payload_len: usize,
    header_len: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PayloadStorage {
    Raw,
    #[cfg(any(feature = "value-compression-lz4", feature = "value-compression-zstd"))]
    Compressed,
}

impl PayloadFrame {
    pub(super) fn raw_without_header(payload_len: usize) -> Self {
        Self {
            storage: PayloadStorage::Raw,
            payload_len,
            header_len: 0,
        }
    }

    pub(crate) fn frame_len(self) -> usize {
        self.header_len + self.payload_len
    }

    pub(crate) fn payload_range(self) -> std::ops::Range<usize> {
        self.header_len..self.header_len + self.payload_len
    }

    pub(crate) fn is_raw_borrowable(self) -> bool {
        self.storage == PayloadStorage::Raw
    }

    pub(super) fn storage(self) -> PayloadStorage {
        self.storage
    }
}

#[cfg(any(feature = "value-compression-lz4", feature = "value-compression-zstd"))]
pub(super) fn parse(
    bytes: &[u8],
    expected_raw_len: usize,
) -> Result<PayloadFrame, CorruptionError> {
    let (&storage, payload) = bytes.split_first().ok_or(CorruptionError::Block)?;
    let storage = match storage {
        STORAGE_RAW => PayloadStorage::Raw,
        STORAGE_COMPRESSED => PayloadStorage::Compressed,
        _ => return Err(CorruptionError::Block),
    };
    if storage == PayloadStorage::Raw && payload.len() != expected_raw_len {
        return Err(CorruptionError::Block);
    }
    if storage == PayloadStorage::Compressed && expected_raw_len > MAX_DECODED_PAYLOAD_LEN {
        return Err(CorruptionError::Block);
    }
    Ok(PayloadFrame {
        storage,
        payload_len: payload.len(),
        header_len: HEADER_LEN,
    })
}

#[cfg(feature = "value-compression-lz4")]
pub(super) fn encode_lz4(
    raw_payload: &[u8],
    policy: ValuePayloadCompressionPolicy,
    out: &mut Vec<u8>,
) -> Result<PayloadFrame, FormatError> {
    if should_store_raw(raw_payload, policy) {
        return Ok(encode_raw(raw_payload, out));
    }

    let compressed = lz4_flex::block::compress(raw_payload);
    if !should_keep_compressed(raw_payload.len(), compressed.len(), policy) {
        return Ok(encode_raw(raw_payload, out));
    }

    write_storage(PayloadStorage::Compressed, out);
    out.extend_from_slice(&compressed);
    Ok(PayloadFrame {
        storage: PayloadStorage::Compressed,
        payload_len: compressed.len(),
        header_len: HEADER_LEN,
    })
}

#[cfg(feature = "value-compression-zstd")]
pub(super) fn encode_zstd(
    raw_payload: &[u8],
    policy: ValuePayloadCompressionPolicy,
    out: &mut Vec<u8>,
    compressor: &mut zstd::bulk::Compressor<'static>,
    scratch: &mut Vec<u8>,
) -> Result<PayloadFrame, FormatError> {
    if should_store_raw(raw_payload, policy) {
        return Ok(encode_raw(raw_payload, out));
    }

    let bound = zstd::zstd_safe::compress_bound(raw_payload.len());
    if scratch.len() < bound {
        scratch.resize(bound, 0);
    }
    let compressed_len = compressor
        .compress_to_buffer(raw_payload, scratch)
        .map_err(|_| FormatError::limit("zstd payload"))?;
    if !should_keep_compressed(raw_payload.len(), compressed_len, policy) {
        return Ok(encode_raw(raw_payload, out));
    }

    write_storage(PayloadStorage::Compressed, out);
    out.extend_from_slice(&scratch[..compressed_len]);
    Ok(PayloadFrame {
        storage: PayloadStorage::Compressed,
        payload_len: compressed_len,
        header_len: HEADER_LEN,
    })
}

#[cfg(any(feature = "value-compression-lz4", feature = "value-compression-zstd"))]
fn should_store_raw(raw_payload: &[u8], policy: ValuePayloadCompressionPolicy) -> bool {
    raw_payload.len() > MAX_DECODED_PAYLOAD_LEN || raw_payload.len() < policy.min_try_len()
}

#[cfg(any(feature = "value-compression-lz4", feature = "value-compression-zstd"))]
fn should_keep_compressed(
    raw_len: usize,
    compressed_len: usize,
    policy: ValuePayloadCompressionPolicy,
) -> bool {
    if compressed_len >= raw_len {
        return false;
    }
    let saved = raw_len - compressed_len;
    saved.saturating_mul(100) >= raw_len.saturating_mul(usize::from(policy.min_saved_percent()))
}

#[cfg(any(feature = "value-compression-lz4", feature = "value-compression-zstd"))]
fn encode_raw(raw_payload: &[u8], out: &mut Vec<u8>) -> PayloadFrame {
    write_storage(PayloadStorage::Raw, out);
    out.extend_from_slice(raw_payload);
    PayloadFrame {
        storage: PayloadStorage::Raw,
        payload_len: raw_payload.len(),
        header_len: HEADER_LEN,
    }
}

#[cfg(any(feature = "value-compression-lz4", feature = "value-compression-zstd"))]
fn write_storage(storage: PayloadStorage, out: &mut Vec<u8>) {
    out.push(match storage {
        PayloadStorage::Raw => STORAGE_RAW,
        PayloadStorage::Compressed => STORAGE_COMPRESSED,
    });
}

#[cfg(all(
    test,
    any(feature = "value-compression-lz4", feature = "value-compression-zstd")
))]
mod tests {
    use super::*;

    #[test]
    fn rejects_unknown_storage() {
        assert!(matches!(parse(&[2], 0), Err(CorruptionError::Block)));
    }

    #[test]
    fn rejects_raw_payload_with_wrong_length() {
        assert!(matches!(
            parse(&[STORAGE_RAW, b'a', b'b', b'c'], 4),
            Err(CorruptionError::Block)
        ));
    }
}
