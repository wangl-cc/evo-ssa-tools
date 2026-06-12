//! Buffered segment entries used by the store commit path.
//!
//! A `WriteBatch` is not a general write subsystem. It is an in-memory arena of
//! logical segment entries that can be sorted, split into immutable segment
//! chunks, and exposed to the segment encoder without per-entry allocations.

use std::ops::Range;

use crate::{
    error::{InputError, Result},
    format::{
        FormatError, ValueLayout,
        entry::{EntryRef, EntrySource, EntryView},
    },
};

/// Buffered writes for one atomic manifest publish.
#[derive(Clone, Debug, Default)]
pub struct WriteBatch {
    entries: Vec<BufferedEntry>,
    key_bytes: Vec<u8>,
    value_bytes: Vec<u8>,
    pub(crate) bytes: usize,
    sorted: bool,
}

/// Per-entry metadata for data stored in the batch arenas.
///
/// The current segment format needs only key/value byte spans. If the on-disk
/// record gains per-entry metadata later, this type is the place where that
/// metadata should live instead of scattering parallel vectors through the
/// writer.
#[derive(Clone, Copy, Debug)]
struct BufferedEntry {
    key: ByteSpan,
    value: ByteSpan,
}

#[derive(Clone, Copy, Debug)]
struct ByteSpan {
    offset: usize,
    len: usize,
}

impl ByteSpan {
    fn new(offset: usize, len: usize) -> Self {
        Self { offset, len }
    }

    fn get(self, bytes: &[u8]) -> &[u8] {
        &bytes[self.offset..self.offset + self.len]
    }
}

impl WriteBatch {
    /// Copies a key/value pair into this batch.
    pub fn push(&mut self, key: &[u8], value: &[u8]) -> Result<()> {
        let key_offset = self.key_bytes.len();
        let value_offset = self.value_bytes.len();
        self.key_bytes.extend_from_slice(key);
        self.value_bytes.extend_from_slice(value);
        self.push_entry(
            ByteSpan::new(key_offset, key.len()),
            ByteSpan::new(value_offset, value.len()),
        )
    }

    /// Moves an owned key/value pair into this batch arena.
    pub fn push_owned(&mut self, mut key: Vec<u8>, mut value: Vec<u8>) -> Result<()> {
        let key = {
            let offset = self.key_bytes.len();
            let len = key.len();
            self.key_bytes.append(&mut key);
            ByteSpan::new(offset, len)
        };
        let value = {
            let offset = self.value_bytes.len();
            let len = value.len();
            self.value_bytes.append(&mut value);
            ByteSpan::new(offset, len)
        };
        self.push_entry(key, value)
    }

    fn push_entry(&mut self, key: ByteSpan, value: ByteSpan) -> Result<()> {
        let entry_bytes = key
            .len
            .checked_add(value.len)
            .ok_or(FormatError::limit("batch byte length"))?;
        self.bytes = self
            .bytes
            .checked_add(entry_bytes)
            .ok_or(FormatError::limit("batch byte length"))?;
        self.entries.push(BufferedEntry { key, value });
        Ok(())
    }

    /// Marks the batch as already sorted by key.
    ///
    /// The store still validates the sorted claim and sorts if needed, but
    /// this avoids the unconditional sort in the common case.
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

    pub(crate) fn validate_lengths(&self, key_len: usize, value_layout: ValueLayout) -> Result<()> {
        for entry in &self.entries {
            if entry.key.len != key_len {
                return Err(InputError::WrongKeyLength {
                    expected: key_len,
                    actual: entry.key.len,
                }
                .into());
            }
            if let Some(fixed_len) = value_layout.fixed_len()
                && entry.value.len != fixed_len.get() as usize
            {
                return Err(InputError::WrongValueLength {
                    expected: fixed_len.get(),
                    actual: entry.value.len,
                }
                .into());
            }
        }
        Ok(())
    }

    pub(crate) fn sort_if_needed(&mut self) {
        if self.sorted && self.is_sorted_by_key() {
            return;
        }
        let key_bytes = &self.key_bytes;
        self.entries
            .sort_by(|left, right| left.key.get(key_bytes).cmp(right.key.get(key_bytes)));
        self.sorted = true;
    }

    pub(crate) fn has_duplicate_keys(&self) -> bool {
        self.entries
            .windows(2)
            .any(|window| self.key_for(window[0]) == self.key_for(window[1]))
    }

    pub(crate) fn flush_ranges(
        &self,
        key_len: usize,
        max_records: usize,
        max_bytes: usize,
    ) -> Vec<Range<usize>> {
        let mut ranges = Vec::new();
        let mut start = 0usize;
        while start < self.entries.len() {
            let mut end = start;
            let mut bytes = 0usize;
            while end < self.entries.len() {
                let entry_bytes = key_len + self.entries[end].value.len;
                let would_exceed_records = end > start && end - start + 1 > max_records;
                let would_exceed_bytes =
                    end > start && (bytes > max_bytes || entry_bytes > max_bytes - bytes);
                if would_exceed_records || would_exceed_bytes {
                    break;
                }
                bytes += entry_bytes;
                end += 1;
            }
            ranges.push(start..end);
            start = end;
        }
        ranges
    }

    pub(crate) fn view(&self, range: Range<usize>) -> EntryView<'_, Self> {
        EntryView::new(self, range)
    }

    pub(crate) fn key_at(&self, index: usize) -> &[u8] {
        self.key_for(self.entries[index])
    }

    fn is_sorted_by_key(&self) -> bool {
        self.entries
            .windows(2)
            .all(|window| self.key_for(window[0]) <= self.key_for(window[1]))
    }

    fn key_for(&self, entry: BufferedEntry) -> &[u8] {
        entry.key.get(&self.key_bytes)
    }

    fn value_for(&self, entry: BufferedEntry) -> &[u8] {
        entry.value.get(&self.value_bytes)
    }
}

impl EntrySource for WriteBatch {
    fn len(&self) -> usize {
        self.entries.len()
    }

    fn entry(&self, index: usize) -> EntryRef<'_> {
        let entry = self.entries[index];
        EntryRef::new(self.key_for(entry), self.value_for(entry))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod arena_entries {
        use super::*;

        #[test]
        fn sorts_entry_metadata_without_changing_payload_slices() {
            let mut batch = WriteBatch::default();
            batch.push(b"key-2", b"value-2").expect("push should work");
            batch.push(b"key-1", b"value-1").expect("push should work");

            batch
                .validate_lengths(5, ValueLayout::VARIABLE)
                .expect("lengths should match");
            batch.sort_if_needed();

            assert_eq!(batch.entry(0).key(), b"key-1");
            assert_eq!(batch.entry(0).value(), b"value-1");
            assert_eq!(batch.entry(1).key(), b"key-2");
            assert_eq!(batch.entry(1).value(), b"value-2");
            assert!(!batch.has_duplicate_keys());
        }

        #[test]
        fn detects_duplicate_keys_after_sorting() {
            let mut batch = WriteBatch::default();
            batch.push(b"key-2", b"value-a").expect("push should work");
            batch.push(b"key-1", b"value-b").expect("push should work");
            batch.push(b"key-2", b"value-c").expect("push should work");

            batch.sort_if_needed();

            assert!(batch.has_duplicate_keys());
        }
    }

    mod flush_splitting {
        use super::*;

        #[test]
        fn splits_by_record_and_byte_thresholds() {
            let mut batch = WriteBatch::default();
            batch.push(b"key-1", b"aaa").expect("push should work");
            batch.push(b"key-2", b"bbb").expect("push should work");
            batch.push(b"key-3", b"ccc").expect("push should work");

            assert_eq!(batch.flush_ranges(5, 2, usize::MAX), vec![0..2, 2..3]);
            assert_eq!(
                batch.flush_ranges(5, usize::MAX, 10),
                vec![0..1, 1..2, 2..3]
            );
        }
    }
}
