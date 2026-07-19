//! Buffered segment entries used by the store commit path.
//!
//! A [`WriteBatch`] is the caller-facing, unvalidated arena. A
//! [`PreparedBatch`] has been checked against store geometry and has strictly
//! increasing keys, so only that representation can reach segment encoders.

use std::ops::Range;

use crate::{
    error::{InputError, Result},
    limits::MAX_VALUE_LEN,
    record::{EntryRef, EntrySource, EntryView},
    value::ValueLayout,
};

/// Buffered writes for one atomic manifest publish.
#[derive(Clone, Debug, Default)]
pub struct WriteBatch {
    entries: Vec<BufferedEntry>,
    key_bytes: Vec<u8>,
    value_bytes: Vec<u8>,
    byte_len: usize,
}

/// Batch validated against store geometry, sorted, and known to have unique keys.
#[derive(Debug, Default)]
pub(super) struct PreparedBatch {
    batch: WriteBatch,
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
    /// Creates an empty write batch.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Copies a key/value pair into this batch.
    pub fn push(&mut self, key: &[u8], value: &[u8]) {
        let key_offset = self.key_bytes.len();
        let value_offset = self.value_bytes.len();
        self.key_bytes.extend_from_slice(key);
        self.value_bytes.extend_from_slice(value);
        self.push_entry(
            ByteSpan::new(key_offset, key.len()),
            ByteSpan::new(value_offset, value.len()),
        );
    }

    fn push_entry(&mut self, key: ByteSpan, value: ByteSpan) {
        self.entries.push(BufferedEntry { key, value });
        self.byte_len = self
            .byte_len
            .saturating_add(key.len)
            .saturating_add(value.len);
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

    pub(super) fn byte_len(&self) -> usize {
        self.byte_len
    }

    pub(super) fn prepare_for(
        mut self,
        key_len: usize,
        value_layout: ValueLayout,
    ) -> Result<PreparedBatch> {
        self.validate_lengths(key_len, value_layout)?;
        let key_bytes = &self.key_bytes;
        self.entries
            .sort_by(|left, right| left.key.get(key_bytes).cmp(right.key.get(key_bytes)));
        if self.has_duplicate_keys() {
            return Err(InputError::DuplicateKeyInBatch.into());
        }
        Ok(PreparedBatch { batch: self })
    }

    fn validate_lengths(&self, key_len: usize, value_layout: ValueLayout) -> Result<()> {
        for entry in &self.entries {
            if entry.key.len != key_len {
                return Err(InputError::WrongKeyLength {
                    expected: key_len,
                    actual: entry.key.len,
                }
                .into());
            }
            if entry.value.len > MAX_VALUE_LEN {
                return Err(InputError::ValueTooLarge {
                    max: MAX_VALUE_LEN,
                    actual: entry.value.len,
                }
                .into());
            }
            if let Some(fixed_len) = value_layout.fixed_value_len()
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

    fn has_duplicate_keys(&self) -> bool {
        self.entries
            .windows(2)
            .any(|window| self.key_for(window[0]) == self.key_for(window[1]))
    }

    fn has_strictly_increasing_keys(&self) -> bool {
        self.entries
            .windows(2)
            .all(|window| self.key_for(window[0]) < self.key_for(window[1]))
    }

    fn key_for(&self, entry: BufferedEntry) -> &[u8] {
        entry.key.get(&self.key_bytes)
    }

    fn value_for(&self, entry: BufferedEntry) -> &[u8] {
        entry.value.get(&self.value_bytes)
    }
}

impl PreparedBatch {
    /// Trust boundary for merge algorithms that emit compatible records in
    /// strictly increasing key order.
    pub(super) fn from_sorted_unique(
        batch: WriteBatch,
        key_len: usize,
        value_layout: ValueLayout,
    ) -> Self {
        debug_assert!(batch.validate_lengths(key_len, value_layout).is_ok());
        debug_assert!(batch.has_strictly_increasing_keys());
        Self { batch }
    }

    pub(super) fn len(&self) -> usize {
        self.batch.len()
    }

    pub(super) fn is_empty(&self) -> bool {
        self.batch.is_empty()
    }

    pub(super) fn byte_len(&self) -> usize {
        self.batch.byte_len()
    }

    pub(super) fn retain_mask(mut self, retain: &[bool]) -> Self {
        assert_eq!(retain.len(), self.batch.entries.len());
        let mut byte_len = 0usize;
        let mut index = 0usize;
        self.batch.entries.retain(|entry| {
            let keep = retain[index];
            index += 1;
            if keep {
                byte_len = byte_len
                    .saturating_add(entry.key.len)
                    .saturating_add(entry.value.len);
            }
            keep
        });
        self.batch.byte_len = byte_len;
        self
    }

    pub(super) fn flush_ranges(
        &self,
        key_len: usize,
        max_records: usize,
        max_bytes: usize,
    ) -> Vec<Range<usize>> {
        let mut ranges = Vec::new();
        let mut start = 0usize;
        while start < self.batch.entries.len() {
            let mut end = start;
            let mut bytes = 0usize;
            while end < self.batch.entries.len() {
                let entry_bytes = key_len + self.batch.entries[end].value.len;
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

    pub(super) fn view(&self, range: Range<usize>) -> EntryView<'_, Self> {
        EntryView::new(self, range)
    }

    pub(super) fn key_at(&self, index: usize) -> &[u8] {
        self.batch.key_for(self.batch.entries[index])
    }
}

impl EntrySource for PreparedBatch {
    fn len(&self) -> usize {
        self.batch.entries.len()
    }

    fn entry(&self, index: usize) -> EntryRef<'_> {
        let entry = self.batch.entries[index];
        EntryRef::new(self.batch.key_for(entry), self.batch.value_for(entry))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod arena_entries {
        use super::*;

        #[test]
        fn sorts_entry_metadata_without_changing_payload_slices() {
            let mut batch = WriteBatch::new();
            batch.push(b"key-2", b"value-2");
            batch.push(b"key-1", b"value-1");

            let batch = batch
                .prepare_for(5, ValueLayout::VARIABLE)
                .expect("batch should prepare");

            assert_eq!(batch.entry(0).key(), b"key-1");
            assert_eq!(batch.entry(0).value(), b"value-1");
            assert_eq!(batch.entry(1).key(), b"key-2");
            assert_eq!(batch.entry(1).value(), b"value-2");
        }

        #[test]
        fn detects_duplicate_keys_after_sorting() {
            let mut batch = WriteBatch::new();
            batch.push(b"key-2", b"value-a");
            batch.push(b"key-1", b"value-b");
            batch.push(b"key-2", b"value-c");

            let error = batch
                .prepare_for(5, ValueLayout::VARIABLE)
                .expect_err("duplicate keys must be rejected");
            assert!(matches!(
                error,
                crate::Error::Input(InputError::DuplicateKeyInBatch)
            ));
        }

        #[test]
        fn rejects_variable_values_above_the_implementation_limit() {
            let batch = WriteBatch {
                entries: vec![BufferedEntry {
                    key: ByteSpan::new(0, 4),
                    value: ByteSpan::new(0, MAX_VALUE_LEN + 1),
                }],
                key_bytes: b"key1".to_vec(),
                value_bytes: Vec::new(),
                byte_len: MAX_VALUE_LEN + 5,
            };

            assert!(matches!(
                batch.validate_lengths(4, ValueLayout::VARIABLE),
                Err(crate::Error::Input(InputError::ValueTooLarge {
                    max: MAX_VALUE_LEN,
                    actual
                })) if actual == MAX_VALUE_LEN + 1
            ));
        }
    }

    mod flush_splitting {
        use super::*;

        #[test]
        fn splits_by_record_and_byte_thresholds() {
            let mut batch = WriteBatch::new();
            batch.push(b"key-1", b"aaa");
            batch.push(b"key-2", b"bbb");
            batch.push(b"key-3", b"ccc");

            let batch = batch
                .prepare_for(5, ValueLayout::VARIABLE)
                .expect("batch should prepare");
            assert_eq!(batch.flush_ranges(5, 2, usize::MAX), vec![0..2, 2..3]);
            assert_eq!(batch.flush_ranges(5, usize::MAX, 10), vec![
                0..1,
                1..2,
                2..3
            ]);
        }
    }
}
