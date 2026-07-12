//! Binary `MANIFEST` snapshot encoding and validation.
//!
//! `MANIFEST` is the atomic visible-data snapshot and the sole source of
//! visibility for a root. This module owns its byte layout and structural
//! invariants (sorted, non-overlapping, unique ids). [`super::paths`] owns file
//! access, while commits construct and publish replacement snapshots.

use std::collections::BTreeSet;

use crc32c::crc32c;

use crate::{
    binary::BinaryCursor,
    error::{CatalogError, CatalogMismatch},
};

const MANIFEST_VERSION: u32 = 1;

const MANIFEST_MAGIC: &[u8; 4] = b"SCSM";
const MANIFEST_HEADER_LEN: usize = 20;
const MANIFEST_TRAILER_LEN: usize = 4;

/// Encoding would exceed the v1 binary `MANIFEST` envelope.
#[derive(thiserror::Error, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ManifestEncodeError {
    #[error("MANIFEST exceeds v1 format limits: entry length overflow")]
    EntryLength,

    #[error("MANIFEST exceeds v1 format limits: length overflow")]
    Length,

    #[error("MANIFEST exceeds v1 format limits: key length exceeds u32")]
    KeyLength,

    #[error("MANIFEST exceeds v1 format limits: segment count exceeds u32")]
    SegmentCount,
}

/// Malformed or corrupt binary `MANIFEST` bytes.
#[derive(thiserror::Error, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ManifestParseError {
    #[error("malformed MANIFEST file: too short")]
    TooShort,

    #[error("malformed MANIFEST file: checksum mismatch")]
    ChecksumMismatch,

    #[error("malformed MANIFEST file: unsupported magic")]
    UnsupportedMagic,

    #[error("malformed MANIFEST file: unsupported segment tier")]
    UnsupportedSegmentTier,

    #[error("malformed MANIFEST file: entry length overflow")]
    EntryLengthOverflow,

    #[error("malformed MANIFEST file: length overflow")]
    LengthOverflow,

    #[error("malformed MANIFEST file: length mismatch")]
    LengthMismatch,

    #[error("malformed MANIFEST file: parser did not consume all entries")]
    ParserDidNotConsumeEntries,

    #[error("malformed MANIFEST file: truncated field {field}")]
    TruncatedField { field: &'static str },
}

/// One visible segment range in `MANIFEST`.
#[derive(Clone, Debug)]
pub(crate) struct SegmentManifestEntry {
    pub(crate) segment_id: u32,
    pub(crate) tier: SegmentTier,
    pub(crate) fingerprint: SegmentFileFingerprint,
    pub(crate) min_key: Vec<u8>,
    pub(crate) max_key: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SegmentFileFingerprint {
    pub(crate) len: u64,
    pub(crate) hash: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SegmentTier {
    Main,
    Patch,
}

impl SegmentManifestEntry {
    pub(crate) fn new(
        segment_id: u32,
        tier: SegmentTier,
        fingerprint: SegmentFileFingerprint,
        min_key: Vec<u8>,
        max_key: Vec<u8>,
    ) -> Self {
        Self {
            segment_id,
            tier,
            fingerprint,
            min_key,
            max_key,
        }
    }

    pub(crate) fn matches_segment(
        &self,
        min_key: &[u8],
        max_key: &[u8],
        fingerprint: SegmentFileFingerprint,
    ) -> bool {
        self.min_key == min_key && self.max_key == max_key && self.fingerprint == fingerprint
    }

    pub(crate) fn is_main(&self) -> bool {
        self.tier == SegmentTier::Main
    }
}

impl SegmentTier {
    fn to_u8(self) -> u8 {
        match self {
            Self::Main => 0,
            Self::Patch => 1,
        }
    }

    fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Main),
            1 => Some(Self::Patch),
            _ => None,
        }
    }
}

/// Parsed `MANIFEST` snapshot.
#[derive(Clone, Debug)]
pub(crate) struct StoreManifest {
    pub(crate) version: u32,
    pub(crate) key_len: usize,
    pub(crate) next_segment_id: u32,
    pub(crate) segments: Vec<SegmentManifestEntry>,
}

impl StoreManifest {
    pub(crate) fn new(key_len: usize) -> Self {
        Self {
            version: MANIFEST_VERSION,
            key_len,
            next_segment_id: 0,
            segments: Vec::new(),
        }
    }

    pub(crate) fn encode(&self) -> std::result::Result<Vec<u8>, ManifestEncodeError> {
        let entry_len = manifest_entry_len(self.key_len).ok_or(ManifestEncodeError::EntryLength)?;
        let capacity = MANIFEST_HEADER_LEN
            .checked_add(
                self.segments
                    .len()
                    .checked_mul(entry_len)
                    .ok_or(ManifestEncodeError::Length)?,
            )
            .and_then(|len| len.checked_add(MANIFEST_TRAILER_LEN))
            .ok_or(ManifestEncodeError::Length)?;
        let mut out = Vec::with_capacity(capacity);
        out.extend_from_slice(MANIFEST_MAGIC);
        out.extend_from_slice(&self.version.to_le_bytes());
        out.extend_from_slice(
            &u32::try_from(self.key_len)
                .map_err(|_| ManifestEncodeError::KeyLength)?
                .to_le_bytes(),
        );
        out.extend_from_slice(&self.next_segment_id.to_le_bytes());
        out.extend_from_slice(
            &u32::try_from(self.segments.len())
                .map_err(|_| ManifestEncodeError::SegmentCount)?
                .to_le_bytes(),
        );
        for entry in &self.segments {
            out.extend_from_slice(&entry.segment_id.to_le_bytes());
            out.push(entry.tier.to_u8());
            out.extend_from_slice(&entry.fingerprint.len.to_le_bytes());
            out.extend_from_slice(&entry.fingerprint.hash.to_le_bytes());
            out.extend_from_slice(&entry.min_key);
            out.extend_from_slice(&entry.max_key);
        }
        let checksum = crc32c(&out);
        out.extend_from_slice(&checksum.to_le_bytes());
        Ok(out)
    }

    pub(crate) fn parse(input: &[u8]) -> std::result::Result<Self, ManifestParseError> {
        ManifestParser::new(input).parse()
    }

    pub(crate) fn validate_structure(
        &self,
        key_len: usize,
    ) -> std::result::Result<(), CatalogError> {
        if self.version != MANIFEST_VERSION {
            return Err(CatalogError::UnsupportedVersion {
                file: "MANIFEST",
                version: self.version,
            });
        }
        if self.key_len != key_len {
            return Err(CatalogMismatch::ManifestKeyLen.into());
        }
        let mut seen_ids = BTreeSet::new();
        let mut max_segment_id: Option<u32> = None;
        let mut previous_max: Option<&[u8]> = None;
        let mut seen_patch = false;
        for entry in &self.segments {
            validate_segment_entry_shape(entry, key_len)?;
            if entry.is_main() {
                if seen_patch {
                    return Err(CatalogMismatch::SegmentTierOrder.into());
                }
                if let Some(previous_max) = previous_max
                    && entry.min_key.as_slice() <= previous_max
                {
                    return Err(CatalogMismatch::SegmentOverlap.into());
                }
                previous_max = Some(&entry.max_key);
            } else {
                seen_patch = true;
            }
            if !seen_ids.insert(entry.segment_id) {
                return Err(CatalogMismatch::DuplicateSegmentId.into());
            }
            max_segment_id =
                Some(max_segment_id.map_or(entry.segment_id, |max| max.max(entry.segment_id)));
        }
        if let Some(max_segment_id) = max_segment_id
            && self.next_segment_id <= max_segment_id
        {
            return Err(CatalogMismatch::NextSegmentId.into());
        }
        Ok(())
    }
}

/// Validates one entry's key shape independent of its neighbors.
///
/// Shared by open-time structure validation above and commit planning.
pub(crate) fn validate_segment_entry_shape(
    entry: &SegmentManifestEntry,
    key_len: usize,
) -> std::result::Result<(), CatalogMismatch> {
    if entry.min_key.len() != key_len || entry.max_key.len() != key_len {
        return Err(CatalogMismatch::SegmentKeyLen);
    }
    if entry.min_key > entry.max_key {
        return Err(CatalogMismatch::SegmentKeyRange);
    }
    Ok(())
}

fn manifest_entry_len(key_len: usize) -> Option<usize> {
    21usize.checked_add(key_len.checked_mul(2)?)
}

struct ManifestParser<'a> {
    input: &'a [u8],
    cursor: BinaryCursor<'a>,
}

impl<'a> ManifestParser<'a> {
    fn new(input: &'a [u8]) -> Self {
        Self {
            input,
            cursor: BinaryCursor::new(input),
        }
    }

    fn parse(mut self) -> std::result::Result<StoreManifest, ManifestParseError> {
        if self.input.len() < MANIFEST_HEADER_LEN + MANIFEST_TRAILER_LEN {
            return Err(ManifestParseError::TooShort);
        }
        let checksum_offset = self.input.len() - MANIFEST_TRAILER_LEN;
        self.cursor.seek(checksum_offset);
        let expected_checksum = self.read_u32("checksum")?;
        let actual_checksum = crc32c(&self.input[..checksum_offset]);
        if actual_checksum != expected_checksum {
            return Err(ManifestParseError::ChecksumMismatch);
        }

        self.cursor.seek(0);
        let magic = self.read_vec("magic", MANIFEST_MAGIC.len())?;
        if magic.as_slice() != MANIFEST_MAGIC {
            return Err(ManifestParseError::UnsupportedMagic);
        }
        let version = self.read_u32("version")?;
        let key_len = self.read_u32("key_len")? as usize;
        let next_segment_id = self.read_u32("next_segment_id")?;
        let segment_count = self.read_u32("segment_count")? as usize;
        let entry_len =
            manifest_entry_len(key_len).ok_or(ManifestParseError::EntryLengthOverflow)?;
        let expected_len = MANIFEST_HEADER_LEN
            .checked_add(
                segment_count
                    .checked_mul(entry_len)
                    .ok_or(ManifestParseError::LengthOverflow)?,
            )
            .and_then(|len| len.checked_add(MANIFEST_TRAILER_LEN))
            .ok_or(ManifestParseError::LengthOverflow)?;
        if self.input.len() != expected_len {
            return Err(ManifestParseError::LengthMismatch);
        }

        let mut segments = Vec::with_capacity(segment_count);
        for _ in 0..segment_count {
            let segment_id = self.read_u32("segment_id")?;
            let tier = SegmentTier::from_u8(self.read_u8("tier")?)
                .ok_or(ManifestParseError::UnsupportedSegmentTier)?;
            let fingerprint = SegmentFileFingerprint {
                len: self.read_u64("segment_len")?,
                hash: self.read_u64("segment_hash")?,
            };
            let min_key = self.read_vec("min_key", key_len)?;
            let max_key = self.read_vec("max_key", key_len)?;
            segments.push(SegmentManifestEntry::new(
                segment_id,
                tier,
                fingerprint,
                min_key,
                max_key,
            ));
        }
        if self.cursor.position() != checksum_offset {
            return Err(ManifestParseError::ParserDidNotConsumeEntries);
        }
        Ok(StoreManifest {
            version,
            key_len,
            next_segment_id,
            segments,
        })
    }

    fn read_u32(&mut self, field: &'static str) -> std::result::Result<u32, ManifestParseError> {
        self.cursor
            .read::<u32>()
            .ok_or(ManifestParseError::TruncatedField { field })
    }

    fn read_u64(&mut self, field: &'static str) -> std::result::Result<u64, ManifestParseError> {
        self.cursor
            .read::<u64>()
            .ok_or(ManifestParseError::TruncatedField { field })
    }

    fn read_u8(&mut self, field: &'static str) -> std::result::Result<u8, ManifestParseError> {
        self.cursor
            .read_slice(1)
            .and_then(|bytes| bytes.first().copied())
            .ok_or(ManifestParseError::TruncatedField { field })
    }

    fn read_vec(
        &mut self,
        field: &'static str,
        len: usize,
    ) -> std::result::Result<Vec<u8>, ManifestParseError> {
        self.cursor
            .read_vec(len)
            .ok_or(ManifestParseError::TruncatedField { field })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(value: u8) -> Vec<u8> {
        vec![value; 16]
    }

    fn manifest_with_segments(segments: Vec<SegmentManifestEntry>) -> StoreManifest {
        StoreManifest {
            version: MANIFEST_VERSION,
            key_len: 16,
            next_segment_id: segments
                .iter()
                .map(|entry| entry.segment_id)
                .max()
                .map_or(0, |max| max + 1),
            segments,
        }
    }

    fn entry(segment_id: u32, min: u8, max: u8) -> SegmentManifestEntry {
        SegmentManifestEntry::new(
            segment_id,
            SegmentTier::Main,
            fingerprint(segment_id),
            key(min),
            key(max),
        )
    }

    fn patch_entry(segment_id: u32, min: u8, max: u8) -> SegmentManifestEntry {
        SegmentManifestEntry::new(
            segment_id,
            SegmentTier::Patch,
            fingerprint(segment_id),
            key(min),
            key(max),
        )
    }

    fn fingerprint(seed: u32) -> SegmentFileFingerprint {
        SegmentFileFingerprint {
            len: 1024 + u64::from(seed),
            hash: u64::from(seed),
        }
    }

    mod manifest_format {
        use super::*;

        #[test]
        fn round_trips_binary_manifest_snapshot() {
            let manifest = manifest_with_segments(vec![entry(7, 0, 1), entry(11, 2, 3)]);

            let bytes = manifest.encode().expect("manifest should encode");
            let parsed = StoreManifest::parse(&bytes).expect("manifest should parse");

            assert_eq!(parsed.version, MANIFEST_VERSION);
            assert_eq!(parsed.key_len, 16);
            assert_eq!(parsed.next_segment_id, 12);
            assert_eq!(parsed.segments.len(), 2);
            assert_eq!(parsed.segments[0].segment_id, 7);
            assert_eq!(parsed.segments[1].min_key, key(2));
        }

        #[test]
        fn rejects_malformed_or_corrupt_binary_manifest() {
            let manifest = manifest_with_segments(vec![entry(0, 0, 1)]);
            let mut bytes = manifest.encode().expect("manifest should encode");
            bytes[0] = b'X';
            assert_eq!(
                StoreManifest::parse(&bytes).unwrap_err(),
                ManifestParseError::ChecksumMismatch
            );

            let mut bytes = manifest.encode().expect("manifest should encode");
            let last = bytes.len() - 1;
            bytes[last] ^= 0xff;
            assert_eq!(
                StoreManifest::parse(&bytes).unwrap_err(),
                ManifestParseError::ChecksumMismatch
            );

            assert_eq!(
                StoreManifest::parse(&[]).unwrap_err(),
                ManifestParseError::TooShort
            );
        }
    }

    mod manifest_validation {
        use super::*;

        #[test]
        fn rejects_duplicate_segment_ids() {
            let duplicate = manifest_with_segments(vec![entry(0, 0, 1), entry(0, 2, 3)]);
            assert!(matches!(
                duplicate.validate_structure(16),
                Err(CatalogError::Mismatch(CatalogMismatch::DuplicateSegmentId))
            ));
        }

        #[test]
        fn rejects_invalid_segment_ranges() {
            let reversed = manifest_with_segments(vec![entry(0, 2, 1)]);
            assert!(matches!(
                reversed.validate_structure(16),
                Err(CatalogError::Mismatch(CatalogMismatch::SegmentKeyRange))
            ));

            let overlapping = manifest_with_segments(vec![entry(0, 0, 2), entry(1, 1, 3)]);
            assert!(matches!(
                overlapping.validate_structure(16),
                Err(CatalogError::Mismatch(CatalogMismatch::SegmentOverlap))
            ));

            let unsorted = manifest_with_segments(vec![entry(1, 2, 3), entry(0, 0, 1)]);
            assert!(matches!(
                unsorted.validate_structure(16),
                Err(CatalogError::Mismatch(CatalogMismatch::SegmentOverlap))
            ));
        }

        #[test]
        fn patch_entries_may_overlap_main_after_main_prefix() {
            let manifest = manifest_with_segments(vec![entry(0, 0, 10), patch_entry(1, 3, 4)]);

            manifest
                .validate_structure(16)
                .expect("patch entries are allowed to overlay main ranges");
        }

        #[test]
        fn patch_entries_must_follow_main_entries() {
            let manifest = manifest_with_segments(vec![patch_entry(1, 3, 4), entry(0, 0, 10)]);

            assert!(matches!(
                manifest.validate_structure(16),
                Err(CatalogError::Mismatch(CatalogMismatch::SegmentTierOrder))
            ));
        }

        #[test]
        fn rejects_reused_next_segment_id() {
            let manifest = StoreManifest {
                version: MANIFEST_VERSION,
                key_len: 16,
                next_segment_id: 1,
                segments: vec![entry(1, 0, 1)],
            };

            assert!(matches!(
                manifest.validate_structure(16),
                Err(CatalogError::Mismatch(CatalogMismatch::NextSegmentId))
            ));
        }
    }
}
