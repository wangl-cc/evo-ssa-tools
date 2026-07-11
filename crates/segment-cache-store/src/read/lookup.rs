//! Ordered lookup implementation and reusable lookup-session state.

use std::sync::Arc;

use crate::{
    engine::runtime::{SegmentSnapshot, SegmentState, StoreGeometry},
    error::{InputError, Result},
    format::{ValuePayloadCompressionKind, ValuePayloadDecoder, block::DecodedBlock},
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

struct LookupState {
    /// The segment snapshot the cached cursor positions below were resolved
    /// against. Cached indices and the loaded block are only meaningful relative
    /// to one snapshot; a commit between calls produces a new snapshot with
    /// shifted segment indices, so the cursor must reset when the snapshot
    /// identity changes.
    snapshot: Option<SegmentSnapshot>,
    current_segment_index: Option<usize>,
    current_block_index: Option<usize>,
    loaded_block: Option<LoadedBlock>,
    payload_decoder: ValuePayloadDecoder,
}

struct LoadedBlock {
    block_index: usize,
    block: DecodedBlock,
    decoded_payload: Option<Vec<u8>>,
}

impl LookupState {
    fn new(value_payload_compression: ValuePayloadCompressionKind) -> Self {
        Self {
            snapshot: None,
            current_segment_index: None,
            current_block_index: None,
            loaded_block: None,
            payload_decoder: ValuePayloadDecoder::new(value_payload_compression),
        }
    }

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
            self.recycle_loaded_block();
        }
    }

    fn recycle_loaded_block(&mut self) -> Vec<u8> {
        let Some(loaded) = self.loaded_block.take() else {
            return Vec::new();
        };
        self.payload_decoder
            .reclaim_payload_buffer(loaded.decoded_payload);
        loaded.block.into_bytes()
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
    pub(crate) geometry: StoreGeometry,
    pub(crate) verify_block_checksums: bool,
}

#[derive(Clone, Copy)]
struct LookupHit<'a> {
    value: &'a [u8],
    segment_index: usize,
    block_index: usize,
    record_index: usize,
}

#[derive(Clone, Copy)]
struct PatchWinner {
    segment_index: usize,
    block_index: usize,
    record_index: usize,
}

impl PatchWinner {
    fn from_hit(hit: LookupHit<'_>) -> Self {
        Self {
            segment_index: hit.segment_index,
            block_index: hit.block_index,
            record_index: hit.record_index,
        }
    }
}

struct PatchWinnerReader<'a> {
    segments: &'a [Arc<SegmentState>],
    states: Vec<Option<LookupState>>,
    options: LookupReadOptions,
}

impl<'a> PatchWinnerReader<'a> {
    fn new(segments: &'a [Arc<SegmentState>], options: LookupReadOptions) -> Self {
        let states = segments.iter().map(|_| None).collect();
        Self {
            segments,
            states,
            options,
        }
    }

    fn should_replace(&mut self, current: Option<PatchWinner>, candidate: &[u8]) -> Result<bool> {
        let Some(current) = current else {
            return Ok(true);
        };
        Ok(self.value(current)?.is_none_or(|winner| candidate < winner))
    }

    fn value(&mut self, winner: PatchWinner) -> Result<Option<&[u8]>> {
        let segment = self
            .segments
            .get(winner.segment_index)
            .expect("patch winner references its source segment");
        let state = self
            .states
            .get_mut(winner.segment_index)
            .and_then(Option::as_mut)
            .expect("patch winner retains its lookup state");
        state.current_segment_index = Some(winner.segment_index);
        match state.value_at_index(
            segment,
            winner.block_index,
            winner.record_index,
            self.options,
        ) {
            Ok(value) => Ok(Some(value)),
            Err(error) if error.is_cache_miss_corruption() => Ok(None),
            Err(error) => Err(error),
        }
    }

    fn keep_state(&mut self, segment_index: usize, state: LookupState) {
        if let Some(slot) = self.states.get_mut(segment_index) {
            *slot = Some(state);
        }
    }
}

impl OrderedLookup {
    pub(crate) fn new(store: Store) -> Self {
        let value_payload_compression = store.inner.geometry.value_payload_compression;
        Self {
            store,
            state: LookupState::new(value_payload_compression),
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
    pub fn visit_many<K, F>(&mut self, keys: &[K], visitor: F) -> Result<()>
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
            let mut visit_value = |index: usize, hit: Option<LookupHit<'_>>| {
                visitor(index, hit.map(|hit| hit.value));
                Ok(())
            };
            OrderedSegmentSweep::new(
                main_segments.as_ref(),
                &mut self.state,
                keys,
                0,
                0,
                options,
                &mut visit_value,
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
            let mut collect_hit = |index: usize, hit: Option<LookupHit<'_>>| {
                if hit.is_some() {
                    results[index] = true;
                }
                Ok(())
            };
            OrderedSegmentSweep::new(
                main_segments.as_ref(),
                &mut self.state,
                keys,
                0,
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
            let mut patch_state = LookupState::new(options.geometry.value_payload_compression);
            let mut collect_hit = |index: usize, hit: Option<LookupHit<'_>>| {
                if hit.is_some() {
                    results[index] = true;
                }
                Ok(())
            };
            OrderedSegmentSweep::new(
                std::slice::from_ref(segment),
                &mut patch_state,
                &keys[key_range.clone()],
                key_range.start,
                0,
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
        let mut patch_reader = PatchWinnerReader::new(patch_segments.as_ref(), options);

        for (patch_segment_index, segment) in patch_segments.iter().enumerate() {
            let key_range = key_range_for_segment(keys, segment);
            if key_range.is_empty() {
                continue;
            }
            let mut patch_state = LookupState::new(options.geometry.value_payload_compression);
            let mut saw_hit = false;
            let mut collect_hit = |index: usize, hit: Option<LookupHit<'_>>| {
                if let Some(hit) = hit {
                    saw_hit = true;
                    let winner = PatchWinner::from_hit(hit);
                    if patch_reader.should_replace(patch_winners[index], hit.value)? {
                        patch_winners[index] = Some(winner);
                    }
                }
                Ok(())
            };
            OrderedSegmentSweep::new(
                std::slice::from_ref(segment),
                &mut patch_state,
                &keys[key_range.clone()],
                key_range.start,
                patch_segment_index,
                options,
                &mut collect_hit,
            )
            .run()?;
            if saw_hit {
                patch_reader.keep_state(patch_segment_index, patch_state);
            }
        }

        self.state.bind_snapshot(main_segments);
        let mut emit_winner = |index: usize, main_hit: Option<LookupHit<'_>>| {
            let main_value = main_hit.map(|hit| hit.value);
            let patch_value = match patch_winners[index] {
                Some(winner) => patch_reader.value(winner)?,
                None => None,
            };
            let winner = match (main_value, patch_value) {
                (Some(main_value), Some(patch_value)) => Some(main_value.min(patch_value)),
                (Some(main_value), None) => Some(main_value),
                (None, Some(patch_value)) => Some(patch_value),
                (None, None) => None,
            };
            visitor(index, winner);
            Ok(())
        };
        OrderedSegmentSweep::new(
            main_segments.as_ref(),
            &mut self.state,
            keys,
            0,
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
        let mut payload_decoder =
            ValuePayloadDecoder::new(self.options.geometry.value_payload_compression);
        let block = match segment.load_block(
            block_index,
            self.options.geometry,
            self.options.verify_block_checksums,
        ) {
            Ok(block) => block,
            Err(error) if error.is_cache_miss_corruption() => return Ok(None),
            Err(error) => return Err(error),
        };
        let record_index = block.lower_bound_index(key);
        if !block.key_matches_at_index(record_index, key) {
            return Ok(None);
        }
        let decoded_payload = match block.decode_payload_if_needed(&mut payload_decoder) {
            Ok(decoded_payload) => decoded_payload,
            Err(_) => return Ok(None),
        };
        Ok(block
            .value_at_index(record_index, decoded_payload.as_deref())
            .map(|value| Some(value.to_vec()))?)
    }
}

struct OrderedSegmentSweep<'a, K, F>
where
    K: AsRef<[u8]>,
    F: FnMut(usize, Option<LookupHit<'_>>) -> Result<()>,
{
    segments: &'a [Arc<SegmentState>],
    lookup_state: &'a mut LookupState,
    keys: &'a [K],
    base_index: usize,
    segment_index_base: usize,
    options: LookupReadOptions,
    visitor: &'a mut F,
    key_index: usize,
    segment_index: usize,
}

impl<'a, K, F> OrderedSegmentSweep<'a, K, F>
where
    K: AsRef<[u8]>,
    F: FnMut(usize, Option<LookupHit<'_>>) -> Result<()>,
{
    fn new(
        segments: &'a [Arc<SegmentState>],
        lookup_state: &'a mut LookupState,
        keys: &'a [K],
        base_index: usize,
        segment_index_base: usize,
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
            segment_index_base,
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
                self.visit_remaining_misses()?;
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
                self.visit_misses_before(segment.min_key.as_slice())?;
                continue;
            }

            let block_index = self.lookup_state.initial_block_index(segment, key);
            let block_end = block_end_from_index(segment, block_index, self.keys, self.key_index);
            if block_end == self.key_index {
                if should_advance_past_empty_block(segment, block_index, key) {
                    self.lookup_state.current_block_index = block_index.checked_add(1);
                    self.lookup_state.recycle_loaded_block();
                } else {
                    (self.visitor)(self.base_index + self.key_index, None)?;
                    self.key_index += 1;
                }
                continue;
            }

            let block =
                match self
                    .lookup_state
                    .ensure_loaded_block(segment, block_index, self.options)
                {
                    Ok(block) => block,
                    Err(error) if error.is_cache_miss_corruption() => {
                        self.visit_corrupt_block_misses(segment, block_index)?;
                        continue;
                    }
                    Err(error) => return Err(error),
                };
            if !BlockKeySweep::new(block, &self.keys[self.key_index..block_end]).has_any_hit() {
                for offset in self.key_index..block_end {
                    (self.visitor)(self.base_index + offset, None)?;
                }
                self.key_index = block_end;
                continue;
            }
            if let Err(error) = self.lookup_state.ensure_decoded_payload() {
                if error.is_cache_miss_corruption() {
                    self.visit_corrupt_block_misses(segment, block_index)?;
                    continue;
                }
                return Err(error);
            }
            let (block, payload) = match self.lookup_state.loaded_block_and_payload() {
                Ok(loaded) => loaded,
                Err(error) if error.is_cache_miss_corruption() => {
                    self.visit_corrupt_block_misses(segment, block_index)?;
                    continue;
                }
                Err(error) => return Err(error),
            };

            BlockKeySweep::new(block, &self.keys[self.key_index..block_end]).visit(
                self.base_index + self.key_index,
                self.segment_index_base + self.segment_index,
                block_index,
                payload,
                self.visitor,
            )?;
            self.key_index = block_end;
        }

        Ok(())
    }

    fn visit_remaining_misses(&mut self) -> Result<()> {
        while self.key_index < self.keys.len() {
            (self.visitor)(self.base_index + self.key_index, None)?;
            self.key_index += 1;
        }
        Ok(())
    }

    fn visit_misses_before(&mut self, min_key: &[u8]) -> Result<()> {
        while self.key_index < self.keys.len() && self.keys[self.key_index].as_ref() < min_key {
            (self.visitor)(self.base_index + self.key_index, None)?;
            self.key_index += 1;
        }
        Ok(())
    }

    fn visit_corrupt_block_misses(
        &mut self,
        segment: &SegmentState,
        block_index: usize,
    ) -> Result<()> {
        let block_end = block_end_from_index(segment, block_index, self.keys, self.key_index);
        for offset in self.key_index..block_end {
            (self.visitor)(self.base_index + offset, None)?;
        }
        self.key_index = block_end;
        self.lookup_state.current_block_index = block_index.checked_add(1);
        self.lookup_state.recycle_loaded_block();
        Ok(())
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
        if let Some(loaded) = self.loaded_block.as_ref()
            && loaded.block.first_key() <= key
            && key <= loaded.block.last_key()
        {
            return loaded.block_index;
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
            self.recycle_loaded_block();
        }
    }

    fn ensure_loaded_block(
        &mut self,
        segment: &SegmentState,
        block_index: usize,
        options: LookupReadOptions,
    ) -> Result<&DecodedBlock> {
        let already_loaded = self
            .loaded_block
            .as_ref()
            .is_some_and(|loaded| loaded.block_index == block_index);
        if already_loaded {
            return Ok(&self
                .loaded_block
                .as_ref()
                .expect("loaded block was just matched")
                .block);
        }

        let buffer = self.recycle_loaded_block();
        let block = segment.load_block_reusing(
            block_index,
            options.geometry,
            options.verify_block_checksums,
            buffer,
        )?;
        self.current_block_index = Some(block_index);
        let loaded = self.loaded_block.insert(LoadedBlock {
            block_index,
            block,
            decoded_payload: None,
        });
        Ok(&loaded.block)
    }

    fn ensure_decoded_payload(&mut self) -> Result<()> {
        let loaded = self
            .loaded_block
            .as_mut()
            .expect("payload decoding follows block loading");
        if loaded.decoded_payload.is_none() {
            loaded.decoded_payload = loaded
                .block
                .decode_payload_if_needed(&mut self.payload_decoder)?;
        }
        Ok(())
    }

    fn loaded_block_and_payload(&self) -> Result<(&DecodedBlock, &[u8])> {
        let loaded = self
            .loaded_block
            .as_ref()
            .expect("payload access follows block loading");
        let payload = loaded
            .block
            .payload_bytes(loaded.decoded_payload.as_deref())?;
        Ok((&loaded.block, payload))
    }

    fn value_at_index(
        &mut self,
        segment: &SegmentState,
        block_index: usize,
        record_index: usize,
        options: LookupReadOptions,
    ) -> Result<&[u8]> {
        self.ensure_loaded_block(segment, block_index, options)?;
        self.ensure_decoded_payload()?;
        let (block, payload) = self.loaded_block_and_payload()?;
        block
            .value_at_index_with_payload(record_index, payload)
            .map_err(Into::into)
    }
}

struct BlockKeySweep<'a, K>
where
    K: AsRef<[u8]>,
{
    block: &'a DecodedBlock,
    keys: &'a [K],
    record_index: usize,
}

impl<'a, K> BlockKeySweep<'a, K>
where
    K: AsRef<[u8]>,
{
    fn new(block: &'a DecodedBlock, keys: &'a [K]) -> Self {
        let record_index = keys
            .first()
            .map_or(0, |key| block.lower_bound_index(key.as_ref()));
        Self {
            block,
            keys,
            record_index,
        }
    }

    fn has_any_hit(mut self) -> bool {
        for key in self.keys {
            if self.key_matches(key.as_ref()) {
                return true;
            }
        }
        false
    }

    fn visit<F>(
        mut self,
        base_index: usize,
        segment_index: usize,
        block_index: usize,
        payload: &'a [u8],
        visitor: &mut F,
    ) -> Result<()>
    where
        F: FnMut(usize, Option<LookupHit<'_>>) -> Result<()>,
    {
        for (offset, key) in self.keys.iter().enumerate() {
            let hit = self
                .value_for(key.as_ref(), payload)
                .map(|(record_index, value)| LookupHit {
                    value,
                    segment_index,
                    block_index,
                    record_index,
                });
            visitor(base_index + offset, hit)?;
        }
        Ok(())
    }

    fn key_matches(&mut self, key: &[u8]) -> bool {
        self.advance_to(key);
        self.block.key_matches_at_index(self.record_index, key)
    }

    fn value_for(&mut self, key: &[u8], payload: &'a [u8]) -> Option<(usize, &'a [u8])> {
        self.advance_to(key);
        let record_index = self.record_index;
        self.block
            .value_at_if_key_with_payload(record_index, key, payload)
            .map(|value| (record_index, value))
    }

    fn advance_to(&mut self, key: &[u8]) {
        while self.record_index < self.block.record_count()
            && self.block.compare_key_at_index(self.record_index, key) == std::cmp::Ordering::Less
        {
            self.record_index += 1;
        }
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

fn should_advance_past_empty_block(segment: &SegmentState, block_index: usize, key: &[u8]) -> bool {
    segment
        .block_index
        .get(block_index + 1)
        .is_some_and(|next| key >= next.first_key.as_slice())
}
