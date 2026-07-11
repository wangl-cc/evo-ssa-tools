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
    engine::{
        StoreStorageStats,
        gc::garbage_collect_unreferenced,
        runtime::{SegmentState, StoreInner},
        storage::collect_storage_stats,
    },
    error::{InputError, Result},
    format::{BlockChecksumKind, StoreMetadata, ValueLayout, ValuePayloadCompressionKind},
    read::{
        cursor::RangeCursor,
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
    // Store identity.

    /// Returns the caller-defined compatibility metadata persisted for this store.
    #[must_use]
    pub fn metadata(&self) -> &StoreMetadata {
        &self.inner.metadata
    }

    /// Returns the fixed key length used by every record in this store.
    #[must_use]
    pub fn key_len(&self) -> usize {
        self.inner.geometry.key_len
    }

    /// Returns the persisted value layout used by this store.
    #[must_use]
    pub fn value_layout(&self) -> ValueLayout {
        self.inner.geometry.value_layout
    }

    /// Returns the block checksum implementation used by this store.
    #[must_use]
    pub fn block_checksum(&self) -> BlockChecksumKind {
        self.inner.geometry.block_checksum
    }

    /// Returns the value-payload compression kind used by this store.
    #[must_use]
    pub fn value_payload_compression(&self) -> ValuePayloadCompressionKind {
        self.inner.geometry.value_payload_compression
    }

    /// Returns physical file usage for this store root.
    ///
    /// This is an operational convenience for tools and diagnostics. It reports
    /// files under the opened root; visible logical record statistics are
    /// intentionally left to read APIs such as [`Store::iter_all`].
    pub fn storage_stats(&self) -> Result<StoreStorageStats> {
        collect_storage_stats(&self.inner.paths)
    }

    // Write path.

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
    /// A batch that interleaves with already-visible main segments is not
    /// rejected. Small interleaving batches are published into a bounded patch
    /// tier first; when the patch tier reaches its bound, the commit normalizes
    /// by merging patch segments, intersecting main segments, and the caller's
    /// batch into replacement main segments. The publication also drops dead
    /// manifest entries (segments that no longer open).
    ///
    /// # Cost of interleaving batches
    ///
    /// Direct main publishes and patch publishes write only the caller's batch.
    /// Normalization rewrites the touched main range plus every live patch
    /// segment, so its cost is driven by key spread and patch-tier occupancy;
    /// [`CommitStats::output_records`] reports the amplification actually paid.
    ///
    /// Returns [`InputError::ReadOnlyStore`] on a read-only handle.
    pub fn commit_batch_with_options(
        &self,
        batch: WriteBatch,
        options: &CommitOptions,
    ) -> Result<CommitStats> {
        self.commit_with_options(batch, options)
    }

    /// Folds all live patch segments into the main tier with default options.
    ///
    /// This is a foreground operation. It is useful after several small
    /// interleaving writes when the caller is about to run a long read-heavy
    /// phase and wants to remove L0 overlay read amplification.
    pub fn normalize(&self) -> Result<CommitStats> {
        self.normalize_with_options(&CommitOptions::default())
    }

    /// Folds all live patch segments into the main tier using explicit segment
    /// write options.
    ///
    /// If no patch segments or dead manifest entries are visible, this is a
    /// no-op and returns default stats. Returns [`InputError::ReadOnlyStore`]
    /// on a read-only handle.
    pub fn normalize_with_options(&self, options: &CommitOptions) -> Result<CommitStats> {
        self.normalize_patches_with_options(options)
    }

    /// Merges another compatible store into this store with default segment write options.
    ///
    /// The destination handle is modified; `source` is only read. Both stores
    /// must have the same caller metadata, key length, and value layout. The
    /// source may use different physical block checksum or value-payload
    /// compression settings because records are decoded and re-written into
    /// the destination format.
    ///
    /// Duplicate keys are resolved with the same deterministic winner rule as
    /// normal commits and range reads: the lexicographically smallest value
    /// bytes survive. The merge is published with one manifest update, so a
    /// failed merge does not make a partial result visible.
    ///
    /// Returns [`InputError::ReadOnlyStore`] on a read-only destination handle.
    pub fn merge_from(&self, source: &Store) -> Result<CommitStats> {
        self.merge_from_with_options(source, &CommitOptions::default())
    }

    /// Merges another compatible store into this store using explicit segment write options.
    ///
    /// `options` controls newly written destination segments only; it does not
    /// have to match the source store's physical format.
    pub fn merge_from_with_options(
        &self,
        source: &Store,
        options: &CommitOptions,
    ) -> Result<CommitStats> {
        self.merge_store_with_options(source, options)
    }

    /// Deletes segment files not referenced by the current manifest.
    ///
    /// Garbage collection is explicit rather than automatic. Commits publish a
    /// new manifest and keep retired segment files in place so a concurrent
    /// read-only open that already read the previous manifest can still open
    /// every file it references. Call this only when the caller accepts that
    /// older manifest snapshots may no longer be openable after the method
    /// returns.
    ///
    /// Returns [`InputError::ReadOnlyStore`] on a read-only handle.
    pub fn garbage_collect(&self) -> Result<()> {
        if self.inner.writer_lock.is_none() {
            return Err(InputError::ReadOnlyStore.into());
        }
        let _commit_guard = self.inner.commit_lock.lock();
        let manifest = {
            let state = self.inner.state.read();
            state.manifest.clone()
        };
        garbage_collect_unreferenced(&self.inner.paths, &manifest)
    }

    // Read path.

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

    /// Visits ordered keys with borrowed value slices.
    pub fn visit_many_ordered<K, F>(&self, keys: &[K], visitor: F) -> Result<()>
    where
        K: AsRef<[u8]>,
        F: FnMut(usize, Option<&[u8]>),
    {
        let mut lookup = self.lookup_session();
        lookup.visit_many(keys, visitor)
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
        let (main_segments, patch_segments) = {
            let state = self.inner.state.read();
            (
                Arc::clone(&state.main_segments),
                Arc::clone(&state.patch_segments),
            )
        };
        SegmentSetReader::new(
            main_segments.as_ref(),
            patch_segments.as_ref(),
            self.lookup_read_options(),
        )
        .fetch_one(key)
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

    // Shared read helpers.

    pub(crate) fn lookup_read_options(&self) -> LookupReadOptions {
        LookupReadOptions {
            geometry: self.inner.geometry,
            verify_block_checksums: self.inner.verify_block_checksums,
        }
    }

    fn range_cursor(&self, start: Option<&[u8]>, end: Option<&[u8]>) -> Result<RangeCursor> {
        let visible = {
            let state = self.inner.state.read();
            (
                Arc::clone(&state.main_segments),
                Arc::clone(&state.patch_segments),
            )
        };
        let (main_segments, patch_segments) = visible;
        let intersects_range = |segment: &&Arc<SegmentState>| {
            if let Some(start) = start
                && segment.max_key.as_slice() < start
            {
                return false;
            }
            if let Some(end) = end
                && segment.min_key.as_slice() >= end
            {
                return false;
            }
            true
        };
        let main_segments = main_segments
            .iter()
            .filter(intersects_range)
            .map(Arc::clone)
            .collect();
        let patch_segments = patch_segments
            .iter()
            .filter(intersects_range)
            .map(Arc::clone)
            .collect();

        Ok(RangeCursor::from_segment_sets(
            main_segments,
            patch_segments,
            self.inner.geometry,
            self.inner.verify_block_checksums,
            start.map(ToOwned::to_owned),
            end.map(ToOwned::to_owned),
        ))
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
