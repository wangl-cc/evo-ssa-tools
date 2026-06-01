//! Public store facade and commit/open orchestration.

use std::{fs, sync::Arc};

use parking_lot::RwLock;

use crate::{
    error::{Error, Result},
    manifest::{
        SegmentFileFingerprint, SegmentManifestEntry, StoreManifest, StorePaths,
        next_segment_file_name, now_unix_millis,
    },
    options::StoreOptions,
    read::{
        cursor::{RangeCursor, SegmentRangeCursor},
        lookup::{OrderedLookup, fetch_from_shard},
    },
    segment::format::{OpenedSegment, SegmentOpenOptions, SegmentWriter, ShardPolicy},
    state::{SegmentState, ShardState, StoreInner, StoreState},
    write::batch::{CommitStats, WriteBatch, flush_ranges, has_duplicate_keys, is_sorted_by_key},
};

/// Persistent append-only cache store.
#[derive(Clone)]
pub struct Store {
    pub(crate) inner: Arc<StoreInner>,
}

impl Store {
    /// Opens an existing store or creates an empty one at `options.root`.
    pub fn open(options: StoreOptions) -> Result<Self> {
        options.validate()?;
        let paths = StorePaths::new(&options.root);
        paths.ensure_dirs(options.shard_count)?;
        let manifest = match StoreManifest::load(&options.root)? {
            Some(manifest) => {
                manifest.validate_options(&options)?;
                manifest
            }
            None => {
                let manifest = StoreManifest::new(&options);
                manifest.store(&options.root)?;
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
                let path = paths.final_segment(shard_id, &entry.file_name);
                let Some(file_len) = SegmentFileFingerprint::read_len_from_path(&path)? else {
                    continue;
                };
                if file_len != entry.file_fingerprint.len {
                    continue;
                }
                let segment = match OpenedSegment::open(path, SegmentOpenOptions {
                    expected_shard: shard_id,
                    expected_key_len: options.key_len,
                    expected_value_layout: options.value_layout,
                    expected_codec_version: options.codec_version,
                })? {
                    Some(segment)
                        if entry.matches_segment_footer(
                            &segment.min_key,
                            &segment.max_key,
                            segment.record_count,
                        ) =>
                    {
                        segment
                    }
                    _ => continue,
                };
                segment_states.push(Arc::new(SegmentState::from_opened(segment)));
            }
            shards.push(ShardState::new(segment_states, last_max_key));
        }

        Ok(Self {
            inner: Arc::new(StoreInner {
                options,
                commit_lock: parking_lot::Mutex::new(()),
                state: RwLock::new(StoreState::new(manifest, shards)),
            }),
        })
    }

    /// Starts a buffered write batch.
    #[must_use]
    pub fn begin_batch(&self) -> WriteBatch {
        WriteBatch::default()
    }

    /// Publishes a write batch by writing immutable segment files and atomically updating the
    /// manifest.
    pub fn commit_batch(&self, batch: WriteBatch) -> Result<CommitStats> {
        if batch.is_empty() {
            return Ok(CommitStats::default());
        }
        let _commit_guard = self.inner.commit_lock.lock();

        let mut sharded = vec![Vec::new(); self.inner.options.shard_count];
        let shard_policy = self.shard_policy();
        for (key, value) in batch.entries {
            self.validate_key_len(&key)?;
            self.validate_value_len(&value)?;
            let shard = shard_policy.shard_for_key(&key);
            sharded[shard].push((key, value));
        }

        let mut stats = CommitStats::default();
        let (mut next_segment_id, mut manifest) = {
            let state = self.inner.state.read();
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
            }
            (state.manifest.next_segment_id, state.manifest.clone())
        };
        let mut new_segments: Vec<Vec<Arc<SegmentState>>> =
            vec![Vec::new(); self.inner.options.shard_count];

        for (shard_id, shard_entries) in sharded.iter_mut().enumerate() {
            if shard_entries.is_empty() {
                continue;
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
                let paths = StorePaths::new(&self.inner.options.root);
                let temp_path = paths.temp_segment(shard_id, segment_id);
                let final_path = paths.final_segment(shard_id, &file_name);
                if final_path.exists() {
                    return Err(Error::SegmentFileAlreadyExists { file_name });
                }
                let (footer, block_index) = SegmentWriter::new(
                    &temp_path,
                    shard_id,
                    self.inner.options.key_len,
                    self.inner.options.value_layout,
                    self.inner.options.codec_version,
                    self.inner.options.target_block_size,
                )
                .write(segment_entries)?;
                let file_fingerprint = SegmentFileFingerprint::read_from_path(&temp_path)?
                    .expect("freshly written segment should exist");
                fs::rename(&temp_path, &final_path)?;
                paths.sync_segment_dir(shard_id)?;
                let file = fs::File::open(&final_path)?;
                let segment_state = Arc::new(SegmentState::from_written(
                    file,
                    footer.min_key.clone(),
                    footer.max_key.clone(),
                    block_index,
                ));
                manifest.shards[shard_id].push(SegmentManifestEntry {
                    file_name,
                    min_key: footer.min_key.clone(),
                    max_key: footer.max_key.clone(),
                    record_count: footer.record_count,
                    created_at_unix_millis: now_unix_millis(),
                    file_fingerprint,
                });
                new_segments[shard_id].push(segment_state);
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
        manifest.store(&self.inner.options.root)?;

        let mut state = self.inner.state.write();
        for (shard_id, shard_new_segments) in new_segments.into_iter().enumerate() {
            if shard_new_segments.is_empty() {
                continue;
            }
            let shard_state = &mut state.shards[shard_id];
            let mut segments = shard_state.segments_as_vec();
            segments.extend(shard_new_segments);
            shard_state.last_max_key = segments.last().map(|segment| segment.max_key.clone());
            shard_state.replace_segments(segments);
        }
        state.refresh_visible_segments();
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

    /// Creates a reusable ordered lookup session with per-shard cursor state.
    #[must_use]
    pub fn lookup_session(&self) -> OrderedLookup {
        OrderedLookup::new(self.clone(), self.inner.options.shard_count)
    }

    /// Fetches one key.
    ///
    /// This exists for completeness; ordered batch lookup is the optimized path.
    pub fn fetch_one(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        self.validate_key_len(key)?;
        let shard_id = self.shard_policy().shard_for_key(key);
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
        let (visible, globally_disjoint) = {
            let state = self.inner.state.read();
            (
                Arc::clone(&state.visible_segments.segments),
                state.visible_segments.globally_disjoint,
            )
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

    fn shard_policy(&self) -> ShardPolicy {
        ShardPolicy::new(
            self.inner.options.shard_count,
            self.inner.options.shard_key_offset,
        )
    }
}
