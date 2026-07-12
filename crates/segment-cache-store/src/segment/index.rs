//! Validated sparse index entries for immutable segment blocks.

use std::{cmp::Ordering, ops::Range};

use crate::key::common_prefix_len;

#[derive(Clone, Debug)]
pub(crate) struct BlockKeyRange {
    encoded: Vec<u8>,
    prefix_len: usize,
    suffix_len: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct BlockIndexEntry {
    pub(crate) key_range: BlockKeyRange,
    pub(crate) byte_range: Range<u64>,
}

impl BlockKeyRange {
    pub(crate) fn new(min_key: &[u8], max_key: &[u8]) -> Option<Self> {
        if min_key.len() != max_key.len() || min_key > max_key {
            return None;
        }
        let prefix_len = common_prefix_len(min_key, max_key);
        let suffix_len = min_key.len() - prefix_len;
        let encoded_len = prefix_len.checked_add(suffix_len.checked_mul(2)?)?;
        let mut encoded = Vec::with_capacity(encoded_len);
        encoded.extend_from_slice(&min_key[..prefix_len]);
        encoded.extend_from_slice(&min_key[prefix_len..]);
        encoded.extend_from_slice(&max_key[prefix_len..]);
        Some(Self {
            encoded,
            prefix_len,
            suffix_len,
        })
    }

    pub(crate) fn from_encoded(
        encoded: Vec<u8>,
        prefix_len: usize,
        key_len: usize,
    ) -> Option<Self> {
        let suffix_len = key_len.checked_sub(prefix_len)?;
        let encoded_len = prefix_len.checked_add(suffix_len.checked_mul(2)?)?;
        if encoded.len() != encoded_len {
            return None;
        }
        let range = Self {
            encoded,
            prefix_len,
            suffix_len,
        };
        if range.min_suffix() > range.max_suffix()
            || common_prefix_len(range.min_suffix(), range.max_suffix()) != 0
        {
            return None;
        }
        Some(range)
    }

    pub(crate) fn prefix(&self) -> &[u8] {
        &self.encoded[..self.prefix_len]
    }

    pub(crate) fn min_suffix(&self) -> &[u8] {
        &self.encoded[self.prefix_len..self.prefix_len + self.suffix_len]
    }

    pub(crate) fn max_suffix(&self) -> &[u8] {
        &self.encoded[self.prefix_len + self.suffix_len..]
    }

    pub(crate) fn min_cmp(&self, key: &[u8]) -> Ordering {
        self.bound_cmp(self.min_suffix(), key)
    }

    pub(crate) fn max_cmp(&self, key: &[u8]) -> Ordering {
        self.bound_cmp(self.max_suffix(), key)
    }

    pub(crate) fn contains(&self, key: &[u8]) -> bool {
        self.min_cmp(key) != Ordering::Greater && self.max_cmp(key) != Ordering::Less
    }

    pub(crate) fn min_equals(&self, key: &[u8]) -> bool {
        self.min_cmp(key) == Ordering::Equal
    }

    pub(crate) fn max_equals(&self, key: &[u8]) -> bool {
        self.max_cmp(key) == Ordering::Equal
    }

    pub(crate) fn ends_before(&self, next: &Self) -> bool {
        self.prefix()
            .iter()
            .chain(self.max_suffix())
            .cmp(next.prefix().iter().chain(next.min_suffix()))
            == Ordering::Less
    }

    fn bound_cmp(&self, suffix: &[u8], key: &[u8]) -> Ordering {
        debug_assert_eq!(self.prefix_len + suffix.len(), key.len());
        match self.prefix().cmp(&key[..self.prefix_len]) {
            Ordering::Equal => suffix.cmp(&key[self.prefix_len..]),
            ordering => ordering,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stores_shared_prefix_once_and_compares_without_materializing_bounds() {
        let range = BlockKeyRange::new(b"shared-001", b"shared-099").expect("valid range");

        assert_eq!(range.prefix(), b"shared-0");
        assert_eq!(range.min_suffix(), b"01");
        assert_eq!(range.max_suffix(), b"99");
        assert_eq!(range.encoded.len(), b"shared-0".len() + 2 * b"01".len());
        assert!(range.contains(b"shared-050"));
        assert!(!range.contains(b"shared-100"));
    }

    #[test]
    fn long_keys_store_one_copy_of_their_shared_prefix() {
        let mut min = vec![b'x'; 512];
        let mut max = min.clone();
        min[510..].copy_from_slice(b"01");
        max[510..].copy_from_slice(b"99");

        let range = BlockKeyRange::new(&min, &max).expect("valid range");

        assert_eq!(range.prefix().len(), 510);
        assert_eq!(range.encoded.len(), 514);
        assert!(range.contains(&[vec![b'x'; 510], b"50".to_vec()].concat()));
    }

    #[test]
    fn rejects_noncanonical_encoded_prefix() {
        assert!(BlockKeyRange::from_encoded(b"aa0a9".to_vec(), 1, 3).is_none());
    }
}
