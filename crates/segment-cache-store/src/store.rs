//! Public store facade.
//!
//! `Store` is a cheaply cloneable shared handle over the engine's runtime
//! state ([`StoreInner`]). This file is the single place that shows the whole
//! operational API — point and range reads, and batched writes — as thin
//! delegators. The heavy algorithms live in their own layers: ordered lookups
//! and range cursors in [`crate::read`], the replacing-manifest commit in
//! [`crate::write`]. Construction (`create`/`open`) lives in [`crate::engine`].

use std::sync::Arc;

use crate::{
    engine::runtime::StoreInner,
    error::{InputError, Result},
    read::{
        cursor::{RangeCursor, SegmentRangeCursor},
        lookup::{LookupReadOptions, OrderedLookup, SegmentSetReader},
    },
    write::{CommitOptions, CommitStats, WriteBatch},
};

/// Persistent append-only cache store.
#[derive(Clone)]
pub struct Store {
    pub(crate) inner: Arc<StoreInner>,
}

impl Store {
    // ─── Write path ─────────────────────────────────────────────────────────

    /// Starts a buffered write batch.
    #[must_use]
    pub fn begin_batch(&self) -> WriteBatch {
        WriteBatch::default()
    }

    /// Publishes a write batch with default segment write options.
    pub fn commit_batch(&self, batch: WriteBatch) -> Result<CommitStats> {
        self.commit_with_options(batch, &CommitOptions::default())
    }

    /// Publishes a write batch using explicit segment write options.
    ///
    /// A batch that interleaves with already-visible segments is not rejected:
    /// the commit rebuilds the intersecting region by merging the batch with the
    /// existing records and publishing replacement segments. The publication also
    /// drops dead manifest entries (segments that no longer open).
    ///
    /// # Cost of interleaving batches
    ///
    /// The rebuilt region is the contiguous run of segments the batch's key
    /// range touches, so the rewrite cost is driven by the batch's **key
    /// spread**, not its size: a two-record batch spanning the whole keyspace
    /// rewrites every visible segment and materializes the merged region in
    /// memory. Until the patch tier lands, callers should keep batches
    /// key-local; [`CommitStats::merged_records`] reports the amplification
    /// actually paid.
    ///
    /// Returns [`InputError::ReadOnlyStore`] on a read-only handle.
    pub fn commit_batch_with_options(
        &self,
        batch: WriteBatch,
        options: &CommitOptions,
    ) -> Result<CommitStats> {
        self.commit_with_options(batch, options)
    }

    /// Flushes pending writes.
    ///
    /// v1 only publishes through `commit_batch`, so this is currently a no-op.
    pub fn flush(&self) -> Result<()> {
        Ok(())
    }

    // ─── Read path ──────────────────────────────────────────────────────────

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

    // ─── Shared read helpers ────────────────────────────────────────────────

    pub(crate) fn lookup_read_options(&self) -> LookupReadOptions {
        LookupReadOptions {
            key_len: self.inner.geometry.key_len,
            value_layout: self.inner.geometry.value_layout,
            verify_block_checksums: self.inner.verify_block_checksums,
        }
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

    fn validate_key_len(&self, key: &[u8]) -> Result<()> {
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
