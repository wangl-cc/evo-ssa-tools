//! Ordered lookup implementation and reusable lookup-session state.

use std::sync::Arc;

use super::LookupReadOptions;
#[cfg(feature = "value-compression")]
use crate::block::ValuePayloadDecoder;
use crate::{
    block::{DecodedBlock, ValuePayloadCompressionKind},
    catalog::SegmentSnapshot,
    error::{InputError, Result},
    key::{BlockRelativeKey, SegmentRelativeKey},
    segment::Segment,
    store::StoreInner,
};

/// Reusable ordered lookup session.
///
/// The session keeps cursor state and the most recently loaded block, making
/// repeated ordered batches cheaper than independent point lookups.
pub struct OrderedLookup {
    inner: Arc<StoreInner>,
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
    #[cfg(feature = "value-compression")]
    payload_decoder: ValuePayloadDecoder,
}

struct LoadedBlock {
    block_index: usize,
    block: DecodedBlock,
    #[cfg(feature = "value-compression")]
    decoded_payload: Option<Vec<u8>>,
}

impl LookupState {
    fn new(_value_payload_compression: ValuePayloadCompressionKind) -> Self {
        Self {
            snapshot: None,
            current_segment_index: None,
            current_block_index: None,
            loaded_block: None,
            #[cfg(feature = "value-compression")]
            payload_decoder: ValuePayloadDecoder::new(_value_payload_compression),
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
        #[cfg(feature = "value-compression")]
        self.payload_decoder
            .reclaim_payload_buffer(loaded.decoded_payload);
        loaded.block.into_bytes()
    }
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
    segments: &'a [Arc<Segment>],
    states: Vec<Option<LookupState>>,
    options: LookupReadOptions,
}

impl<'a> PatchWinnerReader<'a> {
    fn new(segments: &'a [Arc<Segment>], options: LookupReadOptions) -> Self {
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
    pub(crate) fn new(inner: Arc<StoreInner>) -> Self {
        let value_payload_compression = inner.geometry.value_payload_compression;
        Self {
            inner,
            state: LookupState::new(value_payload_compression),
        }
    }

    /// Fetches ordered keys, allocating an owned `Vec<u8>` for each hit.
    pub fn fetch_many<'a, I>(&mut self, keys: I) -> Result<Vec<Option<Vec<u8>>>>
    where
        I: IntoIterator<Item = &'a [u8]>,
    {
        let keys = collect_and_validate_lookup_keys(keys, self.inner.geometry.key_len)?;
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
        let keys = collect_and_validate_lookup_keys(keys, self.inner.geometry.key_len)?;
        self.process_contains_slice(&keys)
    }

    /// Visits ordered keys with borrowed value slices.
    pub fn visit_many<K, F>(&mut self, keys: &[K], visitor: F) -> Result<()>
    where
        K: AsRef<[u8]>,
        F: FnMut(usize, Option<&[u8]>),
    {
        validate_lookup_key_slice(keys, self.inner.geometry.key_len)?;
        self.process_many_slice(keys, visitor)
    }

    fn process_many_slice<K, F>(&mut self, keys: &[K], mut visitor: F) -> Result<()>
    where
        K: AsRef<[u8]>,
        F: FnMut(usize, Option<&[u8]>),
    {
        let options = self.read_options();
        let (main_segments, patch_segments) = {
            let state = self.inner.state.read();
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
        let options = self.read_options();
        let (main_segments, patch_segments) = {
            let state = self.inner.state.read();
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

    fn read_options(&self) -> LookupReadOptions {
        LookupReadOptions {
            geometry: self.inner.geometry,
            verify_block_checksums: self.inner.verify_block_checksums,
        }
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

fn key_range_for_segment<K>(keys: &[K], segment: &Segment) -> std::ops::Range<usize>
where
    K: AsRef<[u8]>,
{
    let start = keys.partition_point(|key| key.as_ref() < segment.min_key());
    let end = keys.partition_point(|key| key.as_ref() <= segment.max_key());
    start..end
}

struct OrderedSegmentSweep<'a, K, F>
where
    K: AsRef<[u8]>,
    F: FnMut(usize, Option<LookupHit<'_>>) -> Result<()>,
{
    segments: &'a [Arc<Segment>],
    lookup_state: &'a mut LookupState,
    keys: &'a [K],
    base_index: usize,
    segment_index_base: usize,
    options: LookupReadOptions,
    visitor: &'a mut F,
    key_index: usize,
    segment_index: usize,
    segment_key_end: usize,
}

impl<'a, K, F> OrderedSegmentSweep<'a, K, F>
where
    K: AsRef<[u8]>,
    F: FnMut(usize, Option<LookupHit<'_>>) -> Result<()>,
{
    fn new(
        segments: &'a [Arc<Segment>],
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
            segment_key_end: 0,
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
            if key > segment.max_key() {
                self.segment_index += 1;
                self.segment_key_end = self.key_index;
                self.lookup_state
                    .reset_segment_if_needed(self.segment_index);
                continue;
            }

            if key < segment.min_key() {
                self.visit_misses_before(segment.min_key())?;
                continue;
            }

            if self.segment_key_end <= self.key_index {
                self.segment_key_end = self.key_index
                    + self.keys[self.key_index..]
                        .partition_point(|candidate| candidate.as_ref() <= segment.max_key());
            }
            let segment_key = segment.relative_key_in_range(key);
            let block_index = self.lookup_state.initial_block_index(segment, segment_key);
            let block_end = block_end_from_index(
                segment,
                block_index,
                self.keys,
                self.key_index,
                self.segment_key_end,
            );
            if block_end == self.key_index {
                (self.visitor)(self.base_index + self.key_index, None)?;
                self.key_index += 1;
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
            let mut block_sweep = BlockKeySweep::new(&self.keys[self.key_index..block_end]);
            if !block_sweep.find_first_hit(block) {
                for offset in self.key_index..block_end {
                    (self.visitor)(self.base_index + offset, None)?;
                }
                self.key_index = block_end;
                continue;
            }
            if let Err(error) = self
                .lookup_state
                .ensure_payload_ready(segment, self.options)
            {
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

            block_sweep.visit_from_first_hit(
                block,
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

    fn visit_corrupt_block_misses(&mut self, segment: &Segment, block_index: usize) -> Result<()> {
        let block_end = block_end_from_index(
            segment,
            block_index,
            self.keys,
            self.key_index,
            self.segment_key_end,
        );
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
    fn initial_segment_index(&self, segments: &[Arc<Segment>], key: &[u8]) -> usize {
        let candidate = self
            .current_segment_index
            .filter(|&index| index < segments.len());
        if let Some(index) = candidate {
            let segment = segments[index].as_ref();
            if segment.min_key() <= key {
                return index;
            }
        }

        segments.partition_point(|segment| segment.max_key() < key)
    }

    fn initial_block_index(&self, segment: &Segment, key: SegmentRelativeKey<'_>) -> usize {
        if let Some(loaded) = self.loaded_block.as_ref()
            && segment.block_contains(loaded.block_index, key)
        {
            return loaded.block_index;
        }

        if let Some(index) = self
            .current_block_index
            .filter(|&index| index < segment.block_count())
            && segment.block_contains(index, key)
        {
            return index;
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
        segment: &Segment,
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
            #[cfg(feature = "value-compression")]
            decoded_payload: None,
        });
        Ok(&loaded.block)
    }

    fn ensure_payload_ready(
        &mut self,
        segment: &Segment,
        options: LookupReadOptions,
    ) -> Result<()> {
        let loaded = self
            .loaded_block
            .as_ref()
            .expect("payload verification follows block loading");
        segment.verify_block_payload(
            loaded.block_index,
            &loaded.block,
            options.verify_block_checksums,
        )?;
        #[cfg(feature = "value-compression")]
        let loaded = self
            .loaded_block
            .as_mut()
            .expect("payload decoding follows block loading");
        #[cfg(feature = "value-compression")]
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
        #[cfg(feature = "value-compression")]
        let payload = loaded
            .block
            .payload_bytes(loaded.decoded_payload.as_deref())?;
        #[cfg(not(feature = "value-compression"))]
        let payload = loaded.block.payload_bytes()?;
        Ok((&loaded.block, payload))
    }

    fn value_at_index(
        &mut self,
        segment: &Segment,
        block_index: usize,
        record_index: usize,
        options: LookupReadOptions,
    ) -> Result<&[u8]> {
        self.ensure_loaded_block(segment, block_index, options)?;
        self.ensure_payload_ready(segment, options)?;
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
    keys: &'a [K],
    query_index: usize,
    record_index: usize,
    initialized: bool,
}

impl<'a, K> BlockKeySweep<'a, K>
where
    K: AsRef<[u8]>,
{
    fn new(keys: &'a [K]) -> Self {
        Self {
            keys,
            query_index: 0,
            record_index: 0,
            initialized: false,
        }
    }

    fn find_first_hit(&mut self, block: &DecodedBlock) -> bool {
        while let Some(key) = self.keys.get(self.query_index) {
            let key = key.as_ref();
            let key = self.advance_to(block, key);
            if block.key_matches_at_index(self.record_index, key) {
                return true;
            }
            self.query_index += 1;
        }
        false
    }

    fn visit_from_first_hit<F>(
        mut self,
        block: &DecodedBlock,
        base_index: usize,
        segment_index: usize,
        block_index: usize,
        payload: &[u8],
        visitor: &mut F,
    ) -> Result<()>
    where
        F: FnMut(usize, Option<LookupHit<'_>>) -> Result<()>,
    {
        for offset in 0..self.query_index {
            visitor(base_index + offset, None)?;
        }

        let first_hit = block
            .value_at_index_with_payload(self.record_index, payload)
            .ok()
            .map(|value| LookupHit {
                value,
                segment_index,
                block_index,
                record_index: self.record_index,
            });
        visitor(base_index + self.query_index, first_hit)?;
        self.query_index += 1;

        while let Some(key) = self.keys.get(self.query_index) {
            let hit = self
                .value_for(block, key.as_ref(), payload)
                .map(|(record_index, value)| LookupHit {
                    value,
                    segment_index,
                    block_index,
                    record_index,
                });
            visitor(base_index + self.query_index, hit)?;
            self.query_index += 1;
        }
        Ok(())
    }

    fn value_for<'value>(
        &mut self,
        block: &DecodedBlock,
        key: &[u8],
        payload: &'value [u8],
    ) -> Option<(usize, &'value [u8])> {
        let key = self.advance_to(block, key);
        let record_index = self.record_index;
        block
            .value_at_if_key_with_payload(record_index, key, payload)
            .map(|value| (record_index, value))
    }

    fn advance_to<'key>(
        &mut self,
        block: &DecodedBlock,
        key: &'key [u8],
    ) -> BlockRelativeKey<'key> {
        let key = block.relative_key_in_range(key);
        if !self.initialized {
            self.record_index = block.lower_bound_index(key);
            self.initialized = true;
            return key;
        }
        while self.record_index < block.record_count()
            && block.compare_key_at_index(self.record_index, key) == std::cmp::Ordering::Less
        {
            self.record_index += 1;
        }
        key
    }
}

fn block_end_from_index<K>(
    segment: &Segment,
    block_index: usize,
    keys: &[K],
    key_index: usize,
    segment_key_end: usize,
) -> usize
where
    K: AsRef<[u8]>,
{
    let mut block_end = key_index;
    while block_end < segment_key_end {
        let key = segment.relative_key_in_range(keys[block_end].as_ref());
        if segment.block_max_cmp(block_index, key) == std::cmp::Ordering::Less {
            break;
        }
        block_end += 1;
    }
    block_end
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, num::NonZeroU32};

    use super::*;
    use crate::{
        block::{
            BlockChecksumKind, BlockDecodeOptions, BlockKeyRangeRef, ValuePayloadCompressionKind,
        },
        value::ValueLayout,
    };

    struct CountedKey<'a> {
        bytes: &'a [u8],
        accesses: Cell<usize>,
    }

    impl AsRef<[u8]> for CountedKey<'_> {
        fn as_ref(&self) -> &[u8] {
            self.accesses.set(self.accesses.get() + 1);
            self.bytes
        }
    }

    #[test]
    fn first_hit_continues_the_same_query_and_record_sweep() -> Result<()> {
        let block = decoded_test_block();
        let keys = [
            CountedKey {
                bytes: b"aa00",
                accesses: Cell::new(0),
            },
            CountedKey {
                bytes: b"aa02",
                accesses: Cell::new(0),
            },
            CountedKey {
                bytes: b"aa03",
                accesses: Cell::new(0),
            },
            CountedKey {
                bytes: b"aa03",
                accesses: Cell::new(0),
            },
        ];
        let mut sweep = BlockKeySweep::new(&keys);
        assert!(sweep.find_first_hit(&block));

        let payload = raw_payload(&block)?;
        let mut visits = Vec::new();
        sweep.visit_from_first_hit(&block, 7, 2, 3, payload, &mut |index, hit| {
            visits.push((index, hit.map(|hit| hit.value.to_vec())));
            Ok(())
        })?;

        assert_eq!(visits, vec![
            (7, None),
            (8, None),
            (9, Some(vec![30])),
            (10, Some(vec![30])),
        ]);
        assert!(keys.iter().all(|key| key.accesses.get() == 1));
        Ok(())
    }

    fn decoded_test_block() -> DecodedBlock {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(b"aa0");
        bytes.extend_from_slice(b"13");
        bytes.extend_from_slice(&[10, 30]);

        DecodedBlock::decode(bytes, BlockDecodeOptions {
            expected_key_range: BlockKeyRangeRef {
                segment_prefix: &[],
                extra_prefix: b"aa0",
                min_suffix: b"1",
                max_suffix: b"3",
            },
            key_len: 4,
            value_layout: ValueLayout::fixed(
                NonZeroU32::new(1).expect("fixed test value width is non-zero"),
            ),
            block_checksum: BlockChecksumKind::None,
            value_payload_compression: ValuePayloadCompressionKind::None,
            #[cfg(feature = "block-checksum")]
            verify_lookup_checksum: true,
        })
        .expect("test block should decode")
    }

    #[cfg(feature = "value-compression")]
    fn raw_payload(block: &DecodedBlock) -> Result<&[u8]> {
        Ok(block.payload_bytes(None)?)
    }

    #[cfg(not(feature = "value-compression"))]
    fn raw_payload(block: &DecodedBlock) -> Result<&[u8]> {
        Ok(block.payload_bytes()?)
    }
}
