//! Public store facade.
//!
//! `Store` is a cheaply cloneable shared handle over the engine's runtime
//! state. This module holds the read-side API surface; the lifecycle
//! operations are implemented next to their mechanics in [`crate::engine`]
//! (`create`, `open`, `commit_batch`).

use std::sync::Arc;

use crate::{
    batch::WriteBatch,
    engine::runtime::StoreInner,
    error::{InputError, Result},
    read::{
        cursor::{RangeCursor, SegmentRangeCursor},
        lookup::{LookupReadOptions, OrderedLookup, SegmentSetReader},
    },
};

/// Persistent append-only cache store.
#[derive(Clone)]
pub struct Store {
    pub(crate) inner: Arc<StoreInner>,
}

impl Store {
    /// Starts a buffered write batch.
    #[must_use]
    pub fn begin_batch(&self) -> WriteBatch {
        WriteBatch::default()
    }

    /// Checks an ordered key stream and returns a cache-safe hit bitmap.
    ///
    /// This is not an index-only probe: normal block checksum verification still
    /// applies, so corrupted blocks report misses.
    pub fn contains_many_ordered<'a, I>(&self, keys: I) -> Result<Vec<bool>>
    where
        I: IntoIterator<Item = &'a [u8]>,
    {
        let mut lookup = self.lookup_session();
        lookup.contains_many(keys)
    }

    /// Fetches an ordered key stream, allocating an owned `Vec<u8>` for each hit.
    pub fn fetch_many_ordered<'a, I>(&self, keys: I) -> Result<Vec<Option<Vec<u8>>>>
    where
        I: IntoIterator<Item = &'a [u8]>,
    {
        let mut lookup = self.lookup_session();
        lookup.fetch_many(keys)
    }

    /// Visits an ordered key stream with borrowed value slices.
    pub fn visit_many_ordered<'a, I, F>(&self, keys: I, visitor: F) -> Result<()>
    where
        I: IntoIterator<Item = &'a [u8]>,
        F: FnMut(usize, Option<&[u8]>),
    {
        let mut lookup = self.lookup_session();
        lookup.visit_many(keys, visitor)
    }

    /// Visits ordered keys already stored in a slice without allocating an intermediate
    /// key-reference list.
    pub fn visit_many_ordered_slice<K, F>(&self, keys: &[K], visitor: F) -> Result<()>
    where
        K: AsRef<[u8]>,
        F: FnMut(usize, Option<&[u8]>),
    {
        let mut lookup = self.lookup_session();
        lookup.visit_many_slice(keys, visitor)
    }

    /// Creates a reusable ordered lookup session with cursor state.
    #[must_use]
    pub fn lookup_session(&self) -> OrderedLookup {
        OrderedLookup::new(self.clone())
    }

    /// Fetches one key.
    ///
    /// This exists for completeness; ordered batch lookup is the optimized path.
    pub fn fetch_one(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        self.validate_key_len(key)?;
        let state = self.inner.state.read();
        SegmentSetReader::new(state.segments.as_ref(), self.lookup_read_options()).fetch_one(key)
    }

    pub(crate) fn lookup_read_options(&self) -> LookupReadOptions {
        LookupReadOptions {
            key_len: self.inner.geometry.key_len,
            value_layout: self.inner.geometry.value_layout,
            verify_block_checksums: self.inner.verify_block_checksums,
        }
    }

    /// Returns an owned iterator over the half-open range `[start, end)`.
    pub fn range(&self, start: &[u8], end: &[u8]) -> Result<RangeCursor> {
        self.validate_key_len(start)?;
        self.validate_key_len(end)?;
        self.range_cursor(Some(start), Some(end))
    }

    /// Visits the half-open range `[start, end)` with borrowed key/value slices.
    pub fn visit_range<F>(&self, start: &[u8], end: &[u8], visitor: F) -> Result<()>
    where
        F: FnMut(&[u8], &[u8]),
    {
        self.validate_key_len(start)?;
        self.validate_key_len(end)?;
        self.range_cursor(Some(start), Some(end))?
            .visit_all(visitor)
    }

    /// Returns an owned iterator over all visible records in key order.
    pub fn iter_all(&self) -> Result<RangeCursor> {
        self.range_cursor(None, None)
    }

    /// Visits all visible records in key order with borrowed key/value slices.
    pub fn visit_all<F>(&self, visitor: F) -> Result<()>
    where
        F: FnMut(&[u8], &[u8]),
    {
        self.range_cursor(None, None)?.visit_all(visitor)
    }

    fn range_cursor(&self, start: Option<&[u8]>, end: Option<&[u8]>) -> Result<RangeCursor> {
        let visible = {
            let state = self.inner.state.read();
            Arc::clone(&state.segments)
        };
        let mut cursors = Vec::with_capacity(visible.len());
        for segment in visible.iter() {
            if let Some(start) = start
                && segment.max_key.as_slice() < start
            {
                continue;
            }
            if let Some(end) = end
                && segment.min_key.as_slice() >= end
            {
                continue;
            }
            cursors.push(SegmentRangeCursor::new(
                Arc::clone(segment),
                self.inner.geometry.key_len,
                self.inner.geometry.value_layout,
                self.inner.verify_block_checksums,
                start.map(ToOwned::to_owned),
                end.map(ToOwned::to_owned),
            )?);
        }

        Ok(RangeCursor::new(cursors))
    }

    /// Flushes pending writes.
    ///
    /// v1 only publishes through `commit_batch`, so this is currently a no-op.
    pub fn flush(&self) -> Result<()> {
        Ok(())
    }

    pub(crate) fn validate_key_len(&self, key: &[u8]) -> Result<()> {
        if key.len() != self.inner.geometry.key_len {
            return Err(InputError::WrongKeyLength {
                expected: self.inner.geometry.key_len,
                actual: key.len(),
            }
            .into());
        }
        Ok(())
    }
}
