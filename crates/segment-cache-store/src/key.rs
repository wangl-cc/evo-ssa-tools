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

    pub(crate) fn strip<'a>(&self, key: &'a [u8]) -> Option<SegmentRelativeKey<'a>> {
        key.strip_prefix(self.as_slice()).map(SegmentRelativeKey)
    }
}

impl<'a> SegmentRelativeKey<'a> {
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

    pub(crate) fn write_to(self, destination: &mut [u8]) -> bool {
        if destination.len()
            != self.segment_prefix.len() + self.extra_prefix.len() + self.suffix.len()
        {
            return false;
        }
        let (segment_prefix, remainder) = destination.split_at_mut(self.segment_prefix.len());
        let (extra_prefix, suffix) = remainder.split_at_mut(self.extra_prefix.len());
        segment_prefix.copy_from_slice(self.segment_prefix);
        extra_prefix.copy_from_slice(self.extra_prefix);
        suffix.copy_from_slice(self.suffix);
        true
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
    fn composite_key_compares_and_materializes_all_parts() {
        let key = CompositeKey::new(b"shared-", b"00", b"7");
        let next = CompositeKey::new(b"shared-", b"00", b"8");
        let mut bytes = [0; 10];

        assert_eq!(key.cmp_slice(b"shared-007"), Ordering::Equal);
        assert_eq!(key.cmp_key(next), Ordering::Less);
        assert!(key.write_to(&mut bytes));
        assert_eq!(&bytes, b"shared-007");
    }

    #[test]
    fn segment_relative_key_requires_the_validated_prefix() {
        let prefix = SegmentKeyPrefix::from_owned(b"shared-".to_vec());

        assert_eq!(
            prefix
                .strip(b"shared-suffix")
                .expect("prefix should match")
                .as_slice(),
            b"suffix"
        );
        assert!(prefix.strip(b"other-suffix").is_none());
    }
}
