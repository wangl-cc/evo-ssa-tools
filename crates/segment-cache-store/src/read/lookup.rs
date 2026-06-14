//! Ordered lookup implementation and reusable lookup-session state.

use std::sync::Arc;

use crate::{
    engine::runtime::{SegmentSnapshot, SegmentState},
    error::{InputError, Result},
    format::{ValueLayout, block::DecodedBlock},
    store::Store,
};

/// Reusable ordered lookup session.
///
/// The session keeps cursor state and the most recently loaded block, making
/// repeated ordered batches cheaper than independent point lookups.
pub struct OrderedLookup {
    store: Store,
    state: LookupState,
}

#[derive(Default)]
struct LookupState {
    /// The segment snapshot the cached cursor positions below were resolved
    /// against. Cached indices and the loaded block are only meaningful relative
    /// to one snapshot; a commit between calls produces a new snapshot with
    /// shifted segment indices, so the cursor must reset when the snapshot
    /// identity changes.
    snapshot: Option<SegmentSnapshot>,
    current_segment_index: Option<usize>,
    current_block_index: Option<usize>,
    loaded_block: Option<DecodedBlock>,
}

impl LookupState {
    /// Rebinds cached cursor state to `snapshot`, discarding stale positions and
    /// the loaded block when the snapshot changed since the previous call.
    fn bind_snapshot(&mut self, snapshot: &SegmentSnapshot) {
        let same = self
            .snapshot
            .as_ref()
            .is_some_and(|current| Arc::ptr_eq(current, snapshot));
        if !same {
            self.snapshot = Some(Arc::clone(snapshot));
            self.current_segment_index = None;
            self.current_block_index = None;
            self.loaded_block = None;
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct SegmentSetReader<'a> {
    main_segments: &'a [Arc<SegmentState>],
    patch_segments: &'a [Arc<SegmentState>],
    options: LookupReadOptions,
}

#[derive(Clone, Copy)]
pub(crate) struct LookupReadOptions {
    pub(crate) key_len: usize,
    pub(crate) value_layout: ValueLayout,
    pub(crate) verify_block_checksums: bool,
}

impl OrderedLookup {
    pub(crate) fn new(store: Store) -> Self {
        Self {
            store,
            state: LookupState::default(),
        }
    }

    /// Fetches ordered keys, allocating an owned `Vec<u8>` for each hit.
    pub fn fetch_many<'a, I>(&mut self, keys: I) -> Result<Vec<Option<Vec<u8>>>>
    where
        I: IntoIterator<Item = &'a [u8]>,
    {
        let keys = collect_and_validate_lookup_keys(keys, self.store.inner.geometry.key_len)?;
        let mut results = Vec::with_capacity(keys.len());
        self.process_many_slice(&keys, |_, value| {
            results.push(value.map(ToOwned::to_owned));
        })?;
        Ok(results)
    }

    /// Checks ordered keys and returns a cache-safe hit bitmap.
    pub fn contains_many<'a, I>(&mut self, keys: I) -> Result<Vec<bool>>
    where
        I: IntoIterator<Item = &'a [u8]>,
    {
        let keys = collect_and_validate_lookup_keys(keys, self.store.inner.geometry.key_len)?;
        self.process_contains_slice(&keys)
    }

    /// Visits ordered keys with borrowed value slices.
    pub fn visit_many<'a, I, F>(&mut self, keys: I, visitor: F) -> Result<()>
    where
        I: IntoIterator<Item = &'a [u8]>,
        F: FnMut(usize, Option<&[u8]>),
    {
        let keys = collect_and_validate_lookup_keys(keys, self.store.inner.geometry.key_len)?;
        self.process_many_slice(&keys, visitor)
    }

    /// Visits ordered keys already stored in a slice.
    pub fn visit_many_slice<K, F>(&mut self, keys: &[K], visitor: F) -> Result<()>
    where
        K: AsRef<[u8]>,
        F: FnMut(usize, Option<&[u8]>),
    {
        validate_lookup_key_slice(keys, self.store.inner.geometry.key_len)?;
        self.process_many_slice(keys, visitor)
    }

    fn process_many_slice<K, F>(&mut self, keys: &[K], mut visitor: F) -> Result<()>
    where
        K: AsRef<[u8]>,
        F: FnMut(usize, Option<&[u8]>),
    {
        let options = self.store.lookup_read_options();
        let (main_segments, patch_segments) = {
            let state = self.store.inner.state.read();
            (state.main_segments.clone(), state.patch_segments.clone())
        };
        if patch_segments.is_empty() {
            self.state.bind_snapshot(&main_segments);
            OrderedSegmentSweep::new(
                main_segments.as_ref(),
                &mut self.state,
                keys,
                0,
                options,
                &mut visitor,
            )
            .run()
        } else {
            self.process_many_with_patches(keys, &main_segments, &patch_segments, options, visitor)
        }
    }

    fn process_contains_slice<K>(&mut self, keys: &[K]) -> Result<Vec<bool>>
    where
        K: AsRef<[u8]>,
    {
        let options = self.store.lookup_read_options();
        let (main_segments, patch_segments) = {
            let state = self.store.inner.state.read();
            (state.main_segments.clone(), state.patch_segments.clone())
        };
        let mut results = vec![false; keys.len()];
        self.state.bind_snapshot(&main_segments);
        {
            let mut collect_hit = |index: usize, value: Option<&[u8]>| {
                if value.is_some() {
                    results[index] = true;
                }
            };
            OrderedSegmentSweep::new(
                main_segments.as_ref(),
                &mut self.state,
                keys,
                0,
                options,
                &mut collect_hit,
            )
            .run()?;
        }

        for segment in patch_segments.iter() {
            let key_range = key_range_for_segment(keys, segment);
            if key_range.is_empty() {
                continue;
            }
            let mut patch_state = LookupState::default();
            let mut collect_hit = |index: usize, value: Option<&[u8]>| {
                if value.is_some() {
                    results[index] = true;
                }
            };
            OrderedSegmentSweep::new(
                std::slice::from_ref(segment),
                &mut patch_state,
                &keys[key_range.clone()],
                key_range.start,
                options,
                &mut collect_hit,
            )
            .run()?;
        }

        Ok(results)
    }

    fn process_many_with_patches<K, F>(
        &mut self,
        keys: &[K],
        main_segments: &SegmentSnapshot,
        patch_segments: &SegmentSnapshot,
        options: LookupReadOptions,
        mut visitor: F,
    ) -> Result<()>
    where
        K: AsRef<[u8]>,
        F: FnMut(usize, Option<&[u8]>),
    {
        let mut patch_winners = vec![None; keys.len()];

        for segment in patch_segments.iter() {
            let key_range = key_range_for_segment(keys, segment);
            if key_range.is_empty() {
                continue;
            }
            let mut patch_state = LookupState::default();
            let mut collect_hit = |index: usize, value: Option<&[u8]>| {
                if let Some(value) = value {
                    keep_winner(&mut patch_winners[index], value);
                }
            };
            OrderedSegmentSweep::new(
                std::slice::from_ref(segment),
                &mut patch_state,
                &keys[key_range.clone()],
                key_range.start,
                options,
                &mut collect_hit,
            )
            .run()?;
        }

        self.state.bind_snapshot(main_segments);
        let mut emit_winner = |index: usize, main_value: Option<&[u8]>| {
            let patch_value = patch_winners[index].as_deref();
            let winner = match (main_value, patch_value) {
                (Some(main_value), Some(patch_value)) => Some(main_value.min(patch_value)),
                (Some(main_value), None) => Some(main_value),
                (None, Some(patch_value)) => Some(patch_value),
                (None, None) => None,
            };
            visitor(index, winner);
        };
        OrderedSegmentSweep::new(
            main_segments.as_ref(),
            &mut self.state,
            keys,
            0,
            options,
            &mut emit_winner,
        )
        .run()?;

        Ok(())
    }
}

fn collect_and_validate_lookup_keys<'a, I>(keys: I, key_len: usize) -> Result<Vec<&'a [u8]>>
where
    I: IntoIterator<Item = &'a [u8]>,
{
    let keys: Vec<_> = keys.into_iter().collect();
    validate_lookup_key_slice(&keys, key_len)?;
    Ok(keys)
}

fn validate_lookup_key_slice<K>(keys: &[K], key_len: usize) -> Result<()>
where
    K: AsRef<[u8]>,
{
    for key in keys {
        let key = key.as_ref();
        if key.len() != key_len {
            return Err(InputError::WrongKeyLength {
                expected: key_len,
                actual: key.len(),
            }
            .into());
        }
    }
    if !keys
        .windows(2)
        .all(|window| window[0].as_ref() <= window[1].as_ref())
    {
        return Err(InputError::UnsortedLookupKeys.into());
    }
    Ok(())
}

fn keep_winner(slot: &mut Option<Vec<u8>>, value: &[u8]) {
    if slot.as_ref().is_none_or(|winner| value < winner.as_slice()) {
        *slot = Some(value.to_vec());
    }
}

fn key_range_for_segment<K>(keys: &[K], segment: &SegmentState) -> std::ops::Range<usize>
where
    K: AsRef<[u8]>,
{
    let start = keys.partition_point(|key| key.as_ref() < segment.min_key.as_slice());
    let end = keys.partition_point(|key| key.as_ref() <= segment.max_key.as_slice());
    start..end
}

impl<'a> SegmentSetReader<'a> {
    pub(crate) fn new(
        main_segments: &'a [Arc<SegmentState>],
        patch_segments: &'a [Arc<SegmentState>],
        options: LookupReadOptions,
    ) -> Self {
        Self {
            main_segments,
            patch_segments,
            options,
        }
    }

    pub(crate) fn fetch_one(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let mut winner = self.fetch_one_from_main(key)?;
        for segment in self.patch_segments {
            if segment.min_key.as_slice() <= key
                && key <= segment.max_key.as_slice()
                && let Some(value) = self.fetch_one_from_segment(segment, key)?
                && winner
                    .as_ref()
                    .is_none_or(|winner| value.as_slice() < winner.as_slice())
            {
                winner = Some(value);
            }
        }
        Ok(winner)
    }

    fn fetch_one_from_main(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let segment_index = match self
            .main_segments
            .partition_point(|segment| segment.min_key.as_slice() <= key)
        {
            0 => return Ok(None),
            idx => idx - 1,
        };
        let segment = &self.main_segments[segment_index];
        if key > segment.max_key.as_slice() {
            return Ok(None);
        }
        self.fetch_one_from_segment(segment, key)
    }

    fn fetch_one_from_segment(
        &self,
        segment: &SegmentState,
        key: &[u8],
    ) -> Result<Option<Vec<u8>>> {
        let block_index = segment.find_block_index(key);
        let block = match segment.load_block(
            block_index,
            self.options.key_len,
            self.options.value_layout,
            self.options.verify_block_checksums,
        ) {
            Ok(block) => block,
            Err(error) if error.is_cache_miss_corruption() => return Ok(None),
            Err(error) => return Err(error),
        };
        Ok(block.find_value(key))
    }
}

struct OrderedSegmentSweep<'a, K, F>
where
    K: AsRef<[u8]>,
    F: FnMut(usize, Option<&[u8]>),
{
    segments: &'a [Arc<SegmentState>],
    lookup_state: &'a mut LookupState,
    keys: &'a [K],
    base_index: usize,
    options: LookupReadOptions,
    visitor: &'a mut F,
    key_index: usize,
    segment_index: usize,
}

impl<'a, K, F> OrderedSegmentSweep<'a, K, F>
where
    K: AsRef<[u8]>,
    F: FnMut(usize, Option<&[u8]>),
{
    fn new(
        segments: &'a [Arc<SegmentState>],
        lookup_state: &'a mut LookupState,
        keys: &'a [K],
        base_index: usize,
        options: LookupReadOptions,
        visitor: &'a mut F,
    ) -> Self {
        let segment_index = keys.first().map_or(0, |key| {
            lookup_state.initial_segment_index(segments, key.as_ref())
        });
        lookup_state.reset_segment_if_needed(segment_index);
        Self {
            segments,
            lookup_state,
            keys,
            base_index,
            options,
            visitor,
            key_index: 0,
            segment_index,
        }
    }

    fn run(&mut self) -> Result<()> {
        while self.key_index < self.keys.len() {
            let key = self.keys[self.key_index].as_ref();

            if self.segment_index >= self.segments.len() {
                self.visit_remaining_misses();
                break;
            }

            let segment = self.segments[self.segment_index].as_ref();
            if key > segment.max_key.as_slice() {
                self.segment_index += 1;
                self.lookup_state
                    .reset_segment_if_needed(self.segment_index);
                continue;
            }

            if key < segment.min_key.as_slice() {
                self.visit_misses_before(segment.min_key.as_slice());
                continue;
            }

            let block_index = self.lookup_state.initial_block_index(segment, key);
            self.lookup_state.current_segment_index = Some(self.segment_index);

            let block =
                match self
                    .lookup_state
                    .ensure_loaded_block(segment, block_index, self.options)
                {
                    Ok(block) => block,
                    Err(error) if error.is_cache_miss_corruption() => {
                        self.visit_corrupt_block_misses(segment, block_index);
                        continue;
                    }
                    Err(error) => return Err(error),
                };
            let block_upper = block.last_key().min(segment.max_key.as_slice());
            let block_end = block_query_end(self.keys, self.key_index, block_upper);
            if block_end == self.key_index {
                if should_advance_past_empty_block(segment, block_index, key) {
                    self.lookup_state.current_block_index = block_index.checked_add(1);
                    self.lookup_state.loaded_block = None;
                } else {
                    (self.visitor)(self.base_index + self.key_index, None);
                    self.key_index += 1;
                }
                continue;
            }

            process_block_keys(
                block,
                &self.keys[self.key_index..block_end],
                self.base_index + self.key_index,
                self.visitor,
            );
            self.key_index = block_end;
        }

        Ok(())
    }

    fn visit_remaining_misses(&mut self) {
        while self.key_index < self.keys.len() {
            (self.visitor)(self.base_index + self.key_index, None);
            self.key_index += 1;
        }
    }

    fn visit_misses_before(&mut self, min_key: &[u8]) {
        while self.key_index < self.keys.len() && self.keys[self.key_index].as_ref() < min_key {
            (self.visitor)(self.base_index + self.key_index, None);
            self.key_index += 1;
        }
    }

    fn visit_corrupt_block_misses(&mut self, segment: &SegmentState, block_index: usize) {
        let block_end = block_end_from_index(segment, block_index, self.keys, self.key_index);
        for offset in self.key_index..block_end {
            (self.visitor)(self.base_index + offset, None);
        }
        self.key_index = block_end;
        self.lookup_state.current_block_index = block_index.checked_add(1);
        self.lookup_state.loaded_block = None;
    }
}

impl LookupState {
    fn initial_segment_index(&self, segments: &[Arc<SegmentState>], key: &[u8]) -> usize {
        let candidate = self
            .current_segment_index
            .filter(|&index| index < segments.len());
        if let Some(index) = candidate {
            let segment = segments[index].as_ref();
            if segment.min_key.as_slice() <= key {
                return index;
            }
        }

        segments.partition_point(|segment| segment.max_key.as_slice() < key)
    }

    fn initial_block_index(&self, segment: &SegmentState, key: &[u8]) -> usize {
        if let Some(block) = self.loaded_block.as_ref()
            && block.first_key() <= key
            && key <= block.last_key()
            && let Some(index) = self.current_block_index
        {
            return index;
        }

        if let Some(index) = self
            .current_block_index
            .filter(|&index| index < segment.block_index.len())
        {
            let entry = &segment.block_index[index];
            let before_next_block = segment
                .block_index
                .get(index + 1)
                .is_none_or(|next| key < next.first_key.as_slice());
            if entry.first_key.as_slice() <= key && before_next_block {
                return index;
            }
        }

        segment.find_block_index(key)
    }

    fn reset_segment_if_needed(&mut self, segment_index: usize) {
        if self.current_segment_index != Some(segment_index) {
            self.current_segment_index = Some(segment_index);
            self.current_block_index = None;
            self.loaded_block = None;
        }
    }

    fn ensure_loaded_block(
        &mut self,
        segment: &SegmentState,
        block_index: usize,
        options: LookupReadOptions,
    ) -> Result<&DecodedBlock> {
        let buffer = match self.loaded_block.take() {
            Some(block) if self.current_block_index == Some(block_index) => {
                return Ok(self.loaded_block.insert(block));
            }
            Some(block) => block.into_bytes(),
            None => Vec::new(),
        };
        let block = segment.load_block_reusing(
            block_index,
            options.key_len,
            options.value_layout,
            options.verify_block_checksums,
            buffer,
        )?;
        self.current_block_index = Some(block_index);
        Ok(self.loaded_block.insert(block))
    }
}

fn block_end_from_index<K>(
    segment: &SegmentState,
    block_index: usize,
    keys: &[K],
    key_index: usize,
) -> usize
where
    K: AsRef<[u8]>,
{
    let next_first_key = segment
        .block_index
        .get(block_index + 1)
        .map(|entry| entry.first_key.as_slice());
    let mut block_end = key_index;
    while block_end < keys.len() {
        let belongs_to_block = match next_first_key {
            Some(next_first_key) => keys[block_end].as_ref() < next_first_key,
            None => keys[block_end].as_ref() <= segment.max_key.as_slice(),
        };
        if !belongs_to_block {
            break;
        }
        block_end += 1;
    }
    block_end
}

fn block_query_end<K>(keys: &[K], key_index: usize, block_upper: &[u8]) -> usize
where
    K: AsRef<[u8]>,
{
    let mut block_end = key_index;
    while block_end < keys.len() && keys[block_end].as_ref() <= block_upper {
        block_end += 1;
    }
    block_end
}

fn should_advance_past_empty_block(segment: &SegmentState, block_index: usize, key: &[u8]) -> bool {
    segment
        .block_index
        .get(block_index + 1)
        .is_some_and(|next| key >= next.first_key.as_slice())
}

fn process_block_keys<K, F>(block: &DecodedBlock, keys: &[K], base_index: usize, visitor: &mut F)
where
    K: AsRef<[u8]>,
    F: FnMut(usize, Option<&[u8]>),
{
    let mut record_index = keys
        .first()
        .map_or(0, |key| block.lower_bound_index(key.as_ref()));

    for (offset, key) in keys.iter().enumerate() {
        let key = key.as_ref();
        while record_index < block.record_count()
            && block.compare_key_at_index(record_index, key) == std::cmp::Ordering::Less
        {
            record_index += 1;
        }

        visitor(
            base_index + offset,
            block.value_at_if_key(record_index, key),
        );
    }
}
