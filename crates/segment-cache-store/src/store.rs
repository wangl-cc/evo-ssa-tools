use std::{fs, sync::Arc};

use parking_lot::RwLock;

use crate::{
    batch::{CommitStats, WriteBatch, flush_ranges, has_duplicate_keys, is_sorted_by_key},
    cursor::{RangeCursor, SegmentRangeCursor},
    error::{Error, Result},
    format::{open_segment, shard_for_key, write_segment},
    lookup::{OrderedLookup, fetch_from_shard},
    manifest::{
        SegmentManifestEntry, StoreManifest, ensure_store_dirs, final_segment_path, load_manifest,
        next_segment_file_name, now_unix_millis, segment_dir, store_manifest, sync_dir,
        temp_segment_path,
    },
    options::StoreOptions,
    state::{ShardState, StoreInner, StoreState, segment_state_from_opened},
};

#[derive(Clone)]
/// Persistent append-only cache store.
pub struct Store {
    pub(crate) inner: Arc<StoreInner>,
}

impl Store {
    /// Opens an existing store or creates an empty one at `options.root`.
    pub fn open(options: StoreOptions) -> Result<Self> {
        options.validate()?;
        ensure_store_dirs(&options.root, options.shard_count)?;
        let manifest = match load_manifest(&options.root)? {
            Some(manifest) => {
                manifest.validate_options(&options)?;
                manifest
            }
            None => {
                let manifest = StoreManifest::new(&options);
                store_manifest(&options.root, &manifest)?;
                manifest
            }
        };

        let mut shards = Vec::with_capacity(options.shard_count);
        for shard_id in 0..options.shard_count {
            let mut segment_states = Vec::new();
            let last_max_key = manifest.shards[shard_id]
                .last()
                .map(|entry| entry.max_key.clone());
            for entry in &manifest.shards[shard_id] {
                let path = final_segment_path(&options.root, shard_id, &entry.file_name);
                if let Some(segment) = open_segment(
                    path,
                    shard_id,
                    options.key_len,
                    options.value_layout,
                    options.codec_version,
                )? {
                    segment_states.push(Arc::new(segment_state_from_opened(segment)));
                }
            }
            shards.push(ShardState {
                segments: segment_states,
                last_max_key,
            });
        }

        Ok(Self {
            inner: Arc::new(StoreInner {
                options,
                state: RwLock::new(StoreState { manifest, shards }),
            }),
        })
    }

    #[must_use]
    /// Starts a buffered write batch.
    pub fn begin_batch(&self) -> WriteBatch {
        WriteBatch::default()
    }

    /// Publishes a write batch by writing immutable segment files and atomically updating the
    /// manifest.
    pub fn commit_batch(&self, batch: WriteBatch) -> Result<CommitStats> {
        if batch.is_empty() {
            return Ok(CommitStats::default());
        }

        let mut sharded = vec![Vec::new(); self.inner.options.shard_count];
        for (key, value) in batch.entries {
            self.validate_key_len(&key)?;
            self.validate_value_len(&value)?;
            let shard = shard_for_key(
                &key,
                self.inner.options.shard_count,
                self.inner.options.shard_key_offset,
            );
            sharded[shard].push((key, value));
        }

        let mut stats = CommitStats::default();
        let mut state = self.inner.state.write();
        let mut next_segment_id = state.manifest.next_segment_id;
        let mut manifest = state.manifest.clone();
        let mut new_segments = Vec::new();

        for (shard_id, shard_entries) in sharded.iter_mut().enumerate() {
            if shard_entries.is_empty() {
                continue;
            }
            if batch.sorted {
                if !is_sorted_by_key(shard_entries) {
                    shard_entries.sort_by(|left, right| left.0.cmp(&right.0));
                }
            } else {
                shard_entries.sort_by(|left, right| left.0.cmp(&right.0));
            }
            if has_duplicate_keys(shard_entries) {
                return Err(Error::DuplicateKeyInBatch);
            }
            if let Some(last_max) = state.shards[shard_id].last_max_key.as_ref()
                && shard_entries[0].0.as_slice() <= last_max.as_slice()
            {
                return Err(Error::OutOfOrderAppend { shard: shard_id });
            }

            for range in flush_ranges(
                shard_entries,
                self.inner.options.key_len,
                self.inner.options.flush_threshold_records,
                self.inner.options.flush_threshold_bytes,
            ) {
                let segment_entries = &shard_entries[range];
                let segment_id = next_segment_id;
                next_segment_id += 1;
                let file_name = next_segment_file_name(segment_id);
                let temp_path = temp_segment_path(&self.inner.options.root, shard_id, segment_id);
                let final_path = final_segment_path(&self.inner.options.root, shard_id, &file_name);
                let (footer, _block_index) = write_segment(
                    &temp_path,
                    shard_id,
                    self.inner.options.key_len,
                    self.inner.options.value_layout,
                    self.inner.options.codec_version,
                    self.inner.options.target_block_size,
                    segment_entries,
                )?;
                fs::rename(&temp_path, &final_path)?;
                sync_dir(&segment_dir(&self.inner.options.root, shard_id))?;
                let opened = open_segment(
                    final_path.clone(),
                    shard_id,
                    self.inner.options.key_len,
                    self.inner.options.value_layout,
                    self.inner.options.codec_version,
                )?
                .expect("freshly written segment should reopen");
                let segment_state = Arc::new(segment_state_from_opened(opened));
                manifest.shards[shard_id].push(SegmentManifestEntry {
                    file_name,
                    min_key: footer.min_key.clone(),
                    max_key: footer.max_key.clone(),
                    record_count: footer.record_count,
                    created_at_unix_millis: now_unix_millis(),
                });
                new_segments.push((shard_id, segment_state, footer.max_key));
                stats.records += segment_entries.len();
                stats.bytes += segment_entries
                    .iter()
                    .map(|(key, value)| key.len() + value.len())
                    .sum::<usize>();
                stats.segments_published += 1;
            }
        }

        manifest.next_segment_id = next_segment_id;
        manifest.target_block_size = self.inner.options.target_block_size;
        store_manifest(&self.inner.options.root, &manifest)?;

        for (shard_id, segment, max_key) in new_segments {
            state.shards[shard_id].segments.push(segment);
            state.shards[shard_id].last_max_key = Some(max_key);
        }
        state.manifest = manifest;
        Ok(stats)
    }

    /// Checks an ordered key stream and returns a hit bitmap.
    pub fn probe_ordered<'a, I>(&self, keys: I) -> Result<Vec<bool>>
    where
        I: IntoIterator<Item = &'a [u8]>,
    {
        let mut lookup = self.lookup_session();
        lookup.probe_many(keys)
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

    #[must_use]
    /// Creates a reusable ordered lookup session with per-shard cursor state.
    pub fn lookup_session(&self) -> OrderedLookup {
        OrderedLookup::new(self.clone(), self.inner.options.shard_count)
    }

    /// Fetches one key.
    ///
    /// This exists for completeness; ordered batch lookup is the optimized path.
    pub fn fetch_one(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        self.validate_key_len(key)?;
        let shard_id = shard_for_key(
            key,
            self.inner.options.shard_count,
            self.inner.options.shard_key_offset,
        );
        let state = self.inner.state.read();
        let shard = &state.shards[shard_id];
        fetch_from_shard(
            shard,
            key,
            self.inner.options.key_len,
            self.inner.options.value_layout,
            self.inner.options.verify_block_checksums,
        )
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
        let state = self.inner.state.read();
        let mut segments = Vec::new();
        for shard in &state.shards {
            for segment in &shard.segments {
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
                segments.push(Arc::clone(segment));
            }
        }
        drop(state);

        segments.sort_by(|left, right| left.min_key.cmp(&right.min_key));
        let globally_disjoint = segments
            .windows(2)
            .all(|window| window[0].max_key.as_slice() < window[1].min_key.as_slice());
        let mut cursors = Vec::with_capacity(segments.len());
        for segment in segments {
            cursors.push(SegmentRangeCursor::new(
                segment,
                self.inner.options.key_len,
                self.inner.options.value_layout,
                self.inner.options.verify_block_checksums,
                start.map(ToOwned::to_owned),
                end.map(ToOwned::to_owned),
            )?);
        }

        Ok(RangeCursor::new(cursors, globally_disjoint))
    }

    /// Flushes pending writes.
    ///
    /// v1 only publishes through `commit_batch`, so this is currently a no-op.
    pub fn flush(&self) -> Result<()> {
        Ok(())
    }

    fn validate_key_len(&self, key: &[u8]) -> Result<()> {
        if key.len() != self.inner.options.key_len {
            return Err(Error::WrongKeyLength {
                expected: self.inner.options.key_len,
                actual: key.len(),
            });
        }
        Ok(())
    }

    fn validate_value_len(&self, value: &[u8]) -> Result<()> {
        if let crate::ValueLayout::Fixed { value_len } = self.inner.options.value_layout
            && value.len() != value_len
        {
            return Err(Error::WrongValueLength {
                expected: value_len,
                actual: value.len(),
            });
        }
        Ok(())
    }
}
