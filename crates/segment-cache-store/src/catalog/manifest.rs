//! Binary `MANIFEST` snapshot encoding and validation.
//!
//! `MANIFEST` is the atomic visible-data snapshot and the sole source of
//! visibility for a root. This module owns its byte layout and structural
//! invariants (sorted, non-overlapping, unique ids). [`super::paths`] owns file
//! access, while commits construct and publish replacement snapshots.

use std::collections::BTreeSet;

use crc32c::crc32c;

use super::{CatalogError, CatalogMismatch};
use crate::{
    binary::BinaryCursor,
    limits::{MAX_KEY_LEN, MAX_MANIFEST_LEN},
    segment::{SEGMENT_CONTENT_ID_LEN, SegmentContentId},
};

const MANIFEST_VERSION: u32 = 1;

const MANIFEST_MAGIC: &[u8; 4] = b"SCSM";
const MANIFEST_HEADER_LEN: usize = 20;
const MANIFEST_TRAILER_LEN: usize = 4;
const MANIFEST_ENTRY_FIXED_LEN: usize = 4 + 1 + 8 + SEGMENT_CONTENT_ID_LEN;

/// Encoding would exceed the v1 binary `MANIFEST` envelope.
#[derive(thiserror::Error, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ManifestEncodeError {
    #[error("MANIFEST exceeds v1 format limits: entry length overflow")]
    EntryLength,

    #[error("MANIFEST exceeds v1 format limits: length overflow")]
    Length,

    #[error("MANIFEST key length exceeds the implementation limit")]
    KeyLength,

    #[error("MANIFEST exceeds v1 format limits: segment count exceeds u32")]
    SegmentCount,

    #[error("MANIFEST exceeds the implementation size limit")]
    TooLarge,
}

/// Malformed or corrupt binary `MANIFEST` bytes.
#[derive(thiserror::Error, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ManifestParseError {
    #[error("malformed MANIFEST file: too short")]
    TooShort,

    #[error("malformed MANIFEST file: exceeds the implementation size limit")]
    TooLarge,

    #[error("malformed MANIFEST file: key length exceeds the implementation limit")]
    KeyLengthTooLarge,

    #[error("malformed MANIFEST file: allocation failed")]
    Allocation,

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
    pub(crate) segment_len: u64,
    pub(crate) content_id: SegmentContentId,
    pub(crate) min_key: Vec<u8>,
    pub(crate) max_key: Vec<u8>,
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
        segment_len: u64,
        content_id: SegmentContentId,
        min_key: Vec<u8>,
        max_key: Vec<u8>,
    ) -> Self {
        Self {
            segment_id,
            tier,
            segment_len,
            content_id,
            min_key,
            max_key,
        }
    }

    pub(crate) fn is_main(&self) -> bool {
        self.tier == SegmentTier::Main
    }

    /// Validates this entry independently of neighboring manifest entries.
    pub(crate) fn validate_shape(&self, key_len: usize) -> Result<(), CatalogMismatch> {
        if self.min_key.len() != key_len || self.max_key.len() != key_len {
            return Err(CatalogMismatch::SegmentKeyLen);
        }
        if self.min_key > self.max_key {
            return Err(CatalogMismatch::SegmentKeyRange);
        }
        Ok(())
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
        if self.key_len > MAX_KEY_LEN {
            return Err(ManifestEncodeError::KeyLength);
        }
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
        if capacity > MAX_MANIFEST_LEN {
            return Err(ManifestEncodeError::TooLarge);
        }
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
            out.extend_from_slice(&entry.segment_len.to_le_bytes());
            out.extend_from_slice(entry.content_id.as_bytes());
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
        let mut previous_patch: Option<(&[u8], u32)> = None;
        let mut seen_patch = false;
        for entry in &self.segments {
            entry.validate_shape(key_len)?;
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
                if previous_patch.is_some_and(|previous| {
                    previous >= (entry.min_key.as_slice(), entry.segment_id)
                }) {
                    return Err(CatalogMismatch::SegmentPatchOrder.into());
                }
                previous_patch = Some((&entry.min_key, entry.segment_id));
            }
            if !seen_ids.insert(entry.segment_id) {
                return Err(CatalogMismatch::DuplicateSegmentId.into());
            }
            max_segment_id =
                Some(max_segment_id.map_or(entry.segment_id, |max| max.max(entry.segment_id)));
        }
        let main_count = self.segments.partition_point(SegmentManifestEntry::is_main);
        let (main_entries, patch_entries) = self.segments.split_at(main_count);
        validate_patch_components(main_entries, patch_entries)?;
        if let Some(max_segment_id) = max_segment_id
            && self.next_segment_id <= max_segment_id
        {
            return Err(CatalogMismatch::NextSegmentId.into());
        }
        Ok(())
    }
}

fn validate_patch_components(
    main_entries: &[SegmentManifestEntry],
    patch_entries: &[SegmentManifestEntry],
) -> std::result::Result<(), CatalogMismatch> {
    let mut main_index = 0;
    let mut patch_index = 0;
    let mut component_max: Option<&[u8]> = None;
    let mut component_has_patch = false;

    while main_index < main_entries.len() || patch_index < patch_entries.len() {
        let take_main = match (main_entries.get(main_index), patch_entries.get(patch_index)) {
            (Some(main), Some(patch)) => main.min_key <= patch.min_key,
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (None, None) => break,
        };
        let (entry, is_patch) = if take_main {
            let entry = &main_entries[main_index];
            main_index += 1;
            (entry, false)
        } else {
            let entry = &patch_entries[patch_index];
            patch_index += 1;
            (entry, true)
        };

        let joins_component = component_max.is_some_and(|max| entry.min_key.as_slice() <= max);
        if !joins_component {
            component_max = Some(&entry.max_key);
            component_has_patch = is_patch;
            continue;
        }
        if is_patch && component_has_patch {
            return Err(CatalogMismatch::MultiplePatchesInComponent);
        }
        component_has_patch |= is_patch;
        if component_max.is_none_or(|max| entry.max_key.as_slice() > max) {
            component_max = Some(&entry.max_key);
        }
    }

    Ok(())
}

fn manifest_entry_len(key_len: usize) -> Option<usize> {
    MANIFEST_ENTRY_FIXED_LEN.checked_add(key_len.checked_mul(2)?)
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
        if self.input.len() > MAX_MANIFEST_LEN {
            return Err(ManifestParseError::TooLarge);
        }
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
        if key_len > MAX_KEY_LEN {
            return Err(ManifestParseError::KeyLengthTooLarge);
        }
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

        let mut segments = Vec::new();
        segments
            .try_reserve_exact(segment_count)
            .map_err(|_| ManifestParseError::Allocation)?;
        for _ in 0..segment_count {
            let segment_id = self.read_u32("segment_id")?;
            let tier = SegmentTier::from_u8(self.read_u8("tier")?)
                .ok_or(ManifestParseError::UnsupportedSegmentTier)?;
            let segment_len = self.read_u64("segment_len")?;
            let content_id = SegmentContentId::from_bytes(
                self.read_array::<SEGMENT_CONTENT_ID_LEN>("segment_content_id")?,
            );
            let min_key = self.read_vec("min_key", key_len)?;
            let max_key = self.read_vec("max_key", key_len)?;
            segments.push(SegmentManifestEntry::new(
                segment_id,
                tier,
                segment_len,
                content_id,
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

    fn read_array<const N: usize>(
        &mut self,
        field: &'static str,
    ) -> std::result::Result<[u8; N], ManifestParseError> {
        self.cursor
            .read_slice(N)
            .and_then(|bytes| bytes.try_into().ok())
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
            1024 + u64::from(segment_id),
            content_id(segment_id),
            key(min),
            key(max),
        )
    }

    fn patch_entry(segment_id: u32, min: u8, max: u8) -> SegmentManifestEntry {
        SegmentManifestEntry::new(
            segment_id,
            SegmentTier::Patch,
            1024 + u64::from(segment_id),
            content_id(segment_id),
            key(min),
            key(max),
        )
    }

    fn content_id(seed: u32) -> SegmentContentId {
        let mut bytes = [0u8; SEGMENT_CONTENT_ID_LEN];
        bytes[..4].copy_from_slice(&seed.to_le_bytes());
        SegmentContentId::from_bytes(bytes)
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
            assert_eq!(parsed.segments[0].segment_len, 1031);
            assert_eq!(parsed.segments[0].content_id, content_id(7));
            assert_eq!(parsed.segments[1].min_key, key(2));
        }

        #[test]
        fn encodes_v1_content_id_at_its_exact_width() {
            let manifest = manifest_with_segments(vec![entry(7, 0, 1)]);
            let bytes = manifest.encode().expect("manifest should encode");
            let entry_start = MANIFEST_HEADER_LEN;

            assert_eq!(
                bytes.len(),
                MANIFEST_HEADER_LEN + MANIFEST_ENTRY_FIXED_LEN + 32 + MANIFEST_TRAILER_LEN
            );
            assert_eq!(
                &bytes[entry_start + 13..entry_start + 45],
                content_id(7).as_bytes()
            );
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
        fn patch_entries_must_use_canonical_key_and_id_order() {
            let descending_key =
                manifest_with_segments(vec![patch_entry(1, 4, 5), patch_entry(2, 3, 4)]);
            assert!(matches!(
                descending_key.validate_structure(16),
                Err(CatalogError::Mismatch(CatalogMismatch::SegmentPatchOrder))
            ));

            let descending_id =
                manifest_with_segments(vec![patch_entry(2, 3, 4), patch_entry(1, 3, 5)]);
            assert!(matches!(
                descending_id.validate_structure(16),
                Err(CatalogError::Mismatch(CatalogMismatch::SegmentPatchOrder))
            ));
        }

        #[test]
        fn rejects_multiple_patches_connected_by_one_main_segment() {
            let manifest = manifest_with_segments(vec![
                entry(0, 0, 10),
                patch_entry(1, 2, 3),
                patch_entry(2, 7, 8),
            ]);

            assert!(matches!(
                manifest.validate_structure(16),
                Err(CatalogError::Mismatch(
                    CatalogMismatch::MultiplePatchesInComponent
                ))
            ));
        }

        #[test]
        fn allows_one_patch_in_each_disjoint_component() {
            let manifest = manifest_with_segments(vec![
                entry(0, 0, 10),
                entry(1, 20, 30),
                patch_entry(2, 2, 3),
                patch_entry(3, 22, 23),
            ]);

            manifest
                .validate_structure(16)
                .expect("disjoint components may each contain one patch");
        }

        #[test]
        fn rejects_directly_overlapping_patch_ranges() {
            let manifest = manifest_with_segments(vec![
                entry(0, 0, 1),
                patch_entry(1, 2, 5),
                patch_entry(2, 4, 7),
            ]);

            assert!(matches!(
                manifest.validate_structure(16),
                Err(CatalogError::Mismatch(
                    CatalogMismatch::MultiplePatchesInComponent
                ))
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
