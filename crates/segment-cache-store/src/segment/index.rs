//! Compact validated sparse index for immutable segment blocks.

use std::{cmp::Ordering, ops::Range, sync::Arc};

use crate::key::{CompositeKey, SegmentKeyPrefix, SegmentRelativeKey, common_prefix_len};

/// One backing allocation containing the segment prefix and every block range.
#[derive(Clone, Debug)]
pub(crate) struct SegmentIndex {
    backing: Arc<[u8]>,
    segment_prefix: SegmentKeyPrefix,
    entries: Vec<BlockIndexEntry>,
}

#[derive(Clone, Debug)]
pub(crate) struct BlockIndexEntry {
    key_range: BlockKeyRange,
    pub(crate) byte_range: Range<u64>,
}

/// Byte ranges parsed from one encoded footer entry before index ownership is established.
pub(super) struct EncodedBlockIndexEntry {
    pub(super) extra_prefix: Range<usize>,
    pub(super) min_suffix: Range<usize>,
    pub(super) max_suffix: Range<usize>,
    pub(super) byte_range: Range<u64>,
}

#[derive(Clone, Debug)]
struct BlockKeyRange {
    extra_prefix: Range<usize>,
    min_suffix: Range<usize>,
    max_suffix: Range<usize>,
}

/// Borrowed physical parts of one validated block range.
#[derive(Clone, Copy)]
pub(crate) struct BlockKeyRangeRef<'a> {
    pub(crate) extra_prefix: &'a [u8],
    pub(crate) min_suffix: &'a [u8],
    pub(crate) max_suffix: &'a [u8],
}

/// Constructs a compact index while a segment is being written.
pub(crate) struct SegmentIndexBuilder {
    key_len: usize,
    segment_prefix_len: usize,
    backing: Vec<u8>,
    entries: Vec<BlockIndexEntry>,
}

impl SegmentIndexBuilder {
    pub(crate) fn new(min_key: &[u8], max_key: &[u8]) -> Option<Self> {
        if min_key.is_empty() || min_key.len() != max_key.len() || min_key > max_key {
            return None;
        }
        let segment_prefix_len = common_prefix_len(min_key, max_key);
        Some(Self {
            key_len: min_key.len(),
            segment_prefix_len,
            backing: min_key[..segment_prefix_len].to_vec(),
            entries: Vec::new(),
        })
    }

    pub(crate) fn segment_prefix_len(&self) -> usize {
        self.segment_prefix_len
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn push(
        &mut self,
        min_key: &[u8],
        max_key: &[u8],
        byte_range: Range<u64>,
    ) -> Option<()> {
        if min_key.len() != self.key_len
            || max_key.len() != self.key_len
            || min_key > max_key
            || min_key[..self.segment_prefix_len] != self.backing[..self.segment_prefix_len]
            || max_key[..self.segment_prefix_len] != self.backing[..self.segment_prefix_len]
        {
            return None;
        }
        let min_relative = &min_key[self.segment_prefix_len..];
        let max_relative = &max_key[self.segment_prefix_len..];
        let extra_prefix_len = common_prefix_len(min_relative, max_relative);
        let extra_prefix = self.append(&min_relative[..extra_prefix_len]);
        let min_suffix = self.append(&min_relative[extra_prefix_len..]);
        let max_suffix = self.append(&max_relative[extra_prefix_len..]);
        self.entries.push(BlockIndexEntry {
            key_range: BlockKeyRange {
                extra_prefix,
                min_suffix,
                max_suffix,
            },
            byte_range,
        });
        Some(())
    }

    pub(crate) fn finish(self) -> SegmentIndex {
        let segment_prefix_range = 0..self.segment_prefix_len;
        let backing: Arc<[u8]> = self.backing.into();
        let segment_prefix =
            SegmentKeyPrefix::from_backing(Arc::clone(&backing), segment_prefix_range)
                .expect("builder owns the segment prefix range");
        SegmentIndex {
            backing,
            segment_prefix,
            entries: self.entries,
        }
    }

    fn append(&mut self, bytes: &[u8]) -> Range<usize> {
        let start = self.backing.len();
        self.backing.extend_from_slice(bytes);
        start..self.backing.len()
    }
}

impl SegmentIndex {
    pub(super) fn from_encoded_parts(
        backing: Arc<[u8]>,
        segment_prefix: Range<usize>,
        entries: Vec<EncodedBlockIndexEntry>,
    ) -> Option<Self> {
        let segment_prefix = SegmentKeyPrefix::from_backing(Arc::clone(&backing), segment_prefix)?;
        let mut block_entries = Vec::new();
        block_entries.try_reserve_exact(entries.len()).ok()?;
        for entry in entries {
            backing.get(entry.extra_prefix.clone())?;
            backing.get(entry.min_suffix.clone())?;
            backing.get(entry.max_suffix.clone())?;
            block_entries.push(BlockIndexEntry {
                key_range: BlockKeyRange {
                    extra_prefix: entry.extra_prefix,
                    min_suffix: entry.min_suffix,
                    max_suffix: entry.max_suffix,
                },
                byte_range: entry.byte_range,
            });
        }
        Some(Self {
            backing,
            segment_prefix,
            entries: block_entries,
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub(crate) fn segment_prefix(&self) -> &SegmentKeyPrefix {
        &self.segment_prefix
    }

    pub(crate) fn entry(&self, index: usize) -> &BlockIndexEntry {
        &self.entries[index]
    }

    pub(crate) fn key_range(&self, index: usize) -> BlockKeyRangeRef<'_> {
        let range = &self.entries[index].key_range;
        BlockKeyRangeRef {
            extra_prefix: &self.backing[range.extra_prefix.clone()],
            min_suffix: &self.backing[range.min_suffix.clone()],
            max_suffix: &self.backing[range.max_suffix.clone()],
        }
    }

    pub(crate) fn contains(&self, index: usize, key: &[u8]) -> bool {
        if let Some(key) = self.segment_prefix.strip(key) {
            return self.relative_min_cmp(index, key) != Ordering::Greater
                && self.relative_max_cmp(index, key) != Ordering::Less;
        }
        self.min_cmp(index, key) != Ordering::Greater && self.max_cmp(index, key) != Ordering::Less
    }

    pub(crate) fn find_block(&self, key: &[u8]) -> usize {
        if let Some(key) = self.segment_prefix.strip(key) {
            return self.find_relative_block(key);
        }
        let mut left = 0;
        let mut right = self.len();
        while left < right {
            let middle = left + (right - left) / 2;
            if self.min_cmp(middle, key) != Ordering::Greater {
                left = middle + 1;
            } else {
                right = middle;
            }
        }
        left.saturating_sub(1)
    }

    fn find_relative_block(&self, key: SegmentRelativeKey<'_>) -> usize {
        let mut left = 0;
        let mut right = self.len();
        while left < right {
            let middle = left + (right - left) / 2;
            if self.relative_min_cmp(middle, key) != Ordering::Greater {
                left = middle + 1;
            } else {
                right = middle;
            }
        }
        left.saturating_sub(1)
    }

    pub(crate) fn min_cmp(&self, index: usize, key: &[u8]) -> Ordering {
        self.bound(index, false).cmp_slice(key)
    }

    pub(crate) fn max_cmp(&self, index: usize, key: &[u8]) -> Ordering {
        self.bound(index, true).cmp_slice(key)
    }

    fn relative_min_cmp(&self, index: usize, key: SegmentRelativeKey<'_>) -> Ordering {
        self.relative_bound(index, false).cmp_slice(key.as_slice())
    }

    fn relative_max_cmp(&self, index: usize, key: SegmentRelativeKey<'_>) -> Ordering {
        self.relative_bound(index, true).cmp_slice(key.as_slice())
    }

    pub(super) fn entries(&self) -> impl Iterator<Item = (&BlockIndexEntry, BlockKeyRangeRef<'_>)> {
        self.entries
            .iter()
            .enumerate()
            .map(|(index, entry)| (entry, self.key_range(index)))
    }

    pub(super) fn validate_ranges(
        &self,
        key_len: usize,
        expected_min_key: &[u8],
        expected_max_key: &[u8],
    ) -> bool {
        if self.is_empty()
            || expected_min_key.len() != key_len
            || expected_max_key.len() != key_len
            || self.segment_prefix.len() > key_len
            || common_prefix_len(expected_min_key, expected_max_key) != self.segment_prefix.len()
            || !expected_min_key.starts_with(self.segment_prefix.as_slice())
        {
            return false;
        }
        for index in 0..self.len() {
            let range = self.key_range(index);
            let relative_len = key_len - self.segment_prefix.len();
            if range.extra_prefix.len() > relative_len
                || range.min_suffix.len() != relative_len - range.extra_prefix.len()
                || range.max_suffix.len() != range.min_suffix.len()
                || range.min_suffix > range.max_suffix
                || common_prefix_len(range.min_suffix, range.max_suffix) != 0
            {
                return false;
            }
        }
        if self.min_cmp(0, expected_min_key) != Ordering::Equal
            || self.max_cmp(self.len() - 1, expected_max_key) != Ordering::Equal
        {
            return false;
        }
        (1..self.len()).all(|index| {
            self.bound(index - 1, true)
                .cmp_key(self.bound(index, false))
                == Ordering::Less
        })
    }

    fn bound(&self, index: usize, max: bool) -> CompositeKey<'_> {
        let range = self.key_range(index);
        CompositeKey::new(
            self.segment_prefix.as_slice(),
            range.extra_prefix,
            if max {
                range.max_suffix
            } else {
                range.min_suffix
            },
        )
    }

    fn relative_bound(&self, index: usize, max: bool) -> CompositeKey<'_> {
        let range = self.key_range(index);
        CompositeKey::new(
            &[],
            range.extra_prefix,
            if max {
                range.max_suffix
            } else {
                range.min_suffix
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stores_segment_prefix_once_and_routes_composite_ranges() {
        let mut min = vec![b'x'; 512];
        let mut max = min.clone();
        min[508..].copy_from_slice(b"0001");
        max[508..].copy_from_slice(b"9999");
        let mut builder = SegmentIndexBuilder::new(&min, &max).expect("valid segment range");
        let mut block_max = min.clone();
        block_max[508..].copy_from_slice(b"0999");
        builder
            .push(&min, &block_max, 20..40)
            .expect("valid block range");
        let mut second_min = min.clone();
        second_min[508..].copy_from_slice(b"9000");
        builder
            .push(&second_min, &max, 40..60)
            .expect("valid second block range");
        let index = builder.finish();

        assert_eq!(index.segment_prefix().len(), 508);
        assert_eq!(index.key_range(0).extra_prefix, b"0");
        assert!(index.contains(0, &min));
        assert!(index.validate_ranges(512, &min, &max));
    }

    #[test]
    fn rejects_nonmaximal_block_prefix() {
        let backing: Arc<[u8]> = b"a00a9".as_slice().into();
        let index = SegmentIndex::from_encoded_parts(backing, 0..0, vec![EncodedBlockIndexEntry {
            extra_prefix: 0..1,
            min_suffix: 1..3,
            max_suffix: 3..5,
            byte_range: 20..40,
        }])
        .expect("ranges are in bounds");

        assert!(!index.validate_ranges(3, b"a00", b"a99"));
    }
}
