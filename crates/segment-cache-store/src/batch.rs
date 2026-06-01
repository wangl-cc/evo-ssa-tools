use std::ops::Range;

use crate::error::Result;

#[derive(Clone, Debug, Default)]
/// Buffered writes for one atomic manifest publish.
pub struct WriteBatch {
    pub(crate) entries: Vec<(Vec<u8>, Vec<u8>)>,
    pub(crate) bytes: usize,
    pub(crate) sorted: bool,
}

impl WriteBatch {
    /// Copies a key/value pair into this batch.
    pub fn push(&mut self, key: &[u8], value: &[u8]) -> Result<()> {
        self.bytes = self
            .bytes
            .saturating_add(key.len())
            .saturating_add(value.len());
        self.entries.push((key.to_vec(), value.to_vec()));
        Ok(())
    }

    /// Moves an owned key/value pair into this batch without copying.
    pub fn push_owned(&mut self, key: Vec<u8>, value: Vec<u8>) -> Result<()> {
        self.bytes = self
            .bytes
            .saturating_add(key.len())
            .saturating_add(value.len());
        self.entries.push((key, value));
        Ok(())
    }

    /// Marks the batch as already sorted by key.
    ///
    /// The store still validates the sorted claim after sharding and sorts if
    /// needed, but this avoids the unconditional sort in the common case.
    #[must_use]
    pub fn mark_sorted(mut self) -> Self {
        self.sorted = true;
        self
    }

    /// Number of records currently buffered.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if the batch contains no records.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
/// Summary returned after a successful batch commit.
pub struct CommitStats {
    /// Records published.
    pub records: usize,
    /// Sum of key and value bytes accepted from the caller.
    pub bytes: usize,
    /// Segment files made visible by the manifest update.
    pub segments_published: usize,
}

pub(crate) fn is_sorted_by_key(entries: &[(Vec<u8>, Vec<u8>)]) -> bool {
    entries.windows(2).all(|window| window[0].0 <= window[1].0)
}

pub(crate) fn has_duplicate_keys(entries: &[(Vec<u8>, Vec<u8>)]) -> bool {
    entries.windows(2).any(|window| window[0].0 == window[1].0)
}

pub(crate) fn flush_ranges(
    entries: &[(Vec<u8>, Vec<u8>)],
    key_len: usize,
    max_records: usize,
    max_bytes: usize,
) -> Vec<Range<usize>> {
    let mut ranges = Vec::new();
    let mut start = 0usize;
    while start < entries.len() {
        let mut end = start;
        let mut bytes = 0usize;
        while end < entries.len() {
            let entry_bytes = key_len.saturating_add(entries[end].1.len());
            let would_exceed_records = end > start && end - start + 1 > max_records;
            let would_exceed_bytes = end > start && bytes.saturating_add(entry_bytes) > max_bytes;
            if would_exceed_records || would_exceed_bytes {
                break;
            }
            bytes = bytes.saturating_add(entry_bytes);
            end += 1;
        }
        ranges.push(start..end);
        start = end;
    }
    ranges
}
