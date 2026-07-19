//! Key-byte representations and algorithms shared by block and segment encoding.

use std::{cmp::Ordering, ops::Range, sync::Arc};

/// Length of the byte prefix shared by `left` and `right`.
pub(crate) fn common_prefix_len(left: &[u8], right: &[u8]) -> usize {
    left.iter()
        .zip(right)
        .take_while(|(left, right)| left == right)
        .count()
}

/// Shared segment-owned prefix context used to interpret relative block keys.
#[derive(Clone, Debug)]
pub(crate) struct SegmentKeyPrefix {
    backing: Arc<[u8]>,
    range: Range<usize>,
}

/// A full query key after one validated segment-prefix strip.
#[derive(Clone, Copy)]
pub(crate) struct SegmentRelativeKey<'a>(&'a [u8]);

/// A segment-relative query key after one validated block-prefix strip.
#[derive(Clone, Copy)]
pub(crate) struct BlockRelativeKey<'a>(&'a [u8]);

impl SegmentKeyPrefix {
    #[cfg(test)]
    pub(crate) fn from_owned(prefix: Vec<u8>) -> Self {
        let len = prefix.len();
        Self {
            backing: prefix.into(),
            range: 0..len,
        }
    }

    pub(crate) fn from_backing(backing: Arc<[u8]>, range: Range<usize>) -> Option<Self> {
        backing.get(range.clone())?;
        Some(Self { backing, range })
    }

    pub(crate) fn as_slice(&self) -> &[u8] {
        &self.backing[self.range.clone()]
    }

    pub(crate) fn len(&self) -> usize {
        self.range.len()
    }

    /// Removes this prefix after the caller has established that `key` lies in
    /// the segment range represented by this prefix.
    pub(crate) fn relative_key<'a>(&self, key: &'a [u8]) -> SegmentRelativeKey<'a> {
        SegmentRelativeKey(&key[self.len()..])
    }
}

impl<'a> SegmentRelativeKey<'a> {
    pub(crate) fn as_slice(self) -> &'a [u8] {
        self.0
    }

    /// Removes a block-owned prefix after block routing established membership.
    pub(crate) fn after_block_prefix(self, prefix_len: usize) -> BlockRelativeKey<'a> {
        BlockRelativeKey(&self.0[prefix_len..])
    }
}

impl<'a> BlockRelativeKey<'a> {
    pub(crate) fn from_suffix(suffix: &'a [u8]) -> Self {
        Self(suffix)
    }

    pub(crate) fn as_slice(self) -> &'a [u8] {
        self.0
    }
}

/// A complete fixed-width key represented by three borrowed physical parts.
#[derive(Clone, Copy)]
pub(crate) struct CompositeKey<'a> {
    segment_prefix: &'a [u8],
    extra_prefix: &'a [u8],
    suffix: &'a [u8],
}

impl<'a> CompositeKey<'a> {
    pub(crate) fn new(segment_prefix: &'a [u8], extra_prefix: &'a [u8], suffix: &'a [u8]) -> Self {
        Self {
            segment_prefix,
            extra_prefix,
            suffix,
        }
    }

    pub(crate) fn cmp_slice(self, key: &[u8]) -> Ordering {
        let encoded_len = self.segment_prefix.len() + self.extra_prefix.len() + self.suffix.len();
        if key.len() != encoded_len {
            return self.bytes().cmp(key.iter());
        }
        let (segment_prefix, relative) = key.split_at(self.segment_prefix.len());
        let ordering = self.segment_prefix.cmp(segment_prefix);
        if ordering != Ordering::Equal {
            return ordering;
        }
        let (extra_prefix, suffix) = relative.split_at(self.extra_prefix.len());
        let ordering = self.extra_prefix.cmp(extra_prefix);
        if ordering != Ordering::Equal {
            return ordering;
        }
        self.suffix.cmp(suffix)
    }

    pub(crate) fn cmp_key(self, other: Self) -> Ordering {
        self.bytes().cmp(other.bytes())
    }

    fn bytes(self) -> impl Iterator<Item = &'a u8> {
        self.segment_prefix
            .iter()
            .chain(self.extra_prefix)
            .chain(self.suffix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn composite_key_compares_all_parts() {
        let key = CompositeKey::new(b"shared-", b"00", b"7");
        let next = CompositeKey::new(b"shared-", b"00", b"8");

        assert_eq!(key.cmp_slice(b"shared-007"), Ordering::Equal);
        assert_eq!(key.cmp_key(next), Ordering::Less);
    }

    #[test]
    fn relative_keys_strip_only_at_validated_range_boundaries() {
        let prefix = SegmentKeyPrefix::from_owned(b"shared-".to_vec());
        let segment_key = prefix.relative_key(b"shared-block-suffix");
        let block_key = segment_key.after_block_prefix(b"block-".len());

        assert_eq!(segment_key.as_slice(), b"block-suffix");
        assert_eq!(block_key.as_slice(), b"suffix");
    }
}
