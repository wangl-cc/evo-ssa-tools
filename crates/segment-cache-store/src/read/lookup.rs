//! Ordered lookup implementation and reusable lookup-session state.

use std::sync::Arc;

use crate::{
    error::{Error, Result},
    options::ValueLayout,
    segment::format::{DecodedBlock, ShardPolicy},
    state::{SegmentState, ShardState},
    store::Store,
};

/// Reusable ordered lookup session.
///
/// The session keeps per-shard cursor state and the most recently loaded block,
/// making repeated ordered batches cheaper than independent point lookups.
pub struct OrderedLookup {
    store: Store,
    shard_states: Vec<ShardLookupState>,
}

#[derive(Default)]
struct ShardLookupState {
    current_segment_index: Option<usize>,
    current_block_index: Option<usize>,
    loaded_block: Option<DecodedBlock>,
}

#[derive(Clone, Copy)]
struct LookupReadOptions {
    key_len: usize,
    value_layout: ValueLayout,
    verify_block_checksums: bool,
}

impl OrderedLookup {
    pub(crate) fn new(store: Store, shard_count: usize) -> Self {
        Self {
            store,
            shard_states: (0..shard_count)
                .map(|_| ShardLookupState::default())
                .collect(),
        }
    }

    /// Fetches ordered keys, allocating an owned `Vec<u8>` for each hit.
    pub fn fetch_many<'a, I>(&mut self, keys: I) -> Result<Vec<Option<Vec<u8>>>>
    where
        I: IntoIterator<Item = &'a [u8]>,
    {
        let keys = collect_and_validate_lookup_keys(keys, self.store.inner.options.key_len)?;
        let mut results = Vec::with_capacity(keys.len());
        self.process_many_slice(&keys, |_, value| {
            results.push(value.map(ToOwned::to_owned));
        })?;
        Ok(results)
    }

    /// Checks ordered keys and returns a hit bitmap.
    pub fn probe_many<'a, I>(&mut self, keys: I) -> Result<Vec<bool>>
    where
        I: IntoIterator<Item = &'a [u8]>,
    {
        let keys = collect_and_validate_lookup_keys(keys, self.store.inner.options.key_len)?;
        let mut results = Vec::with_capacity(keys.len());
        self.process_many_slice(&keys, |_, value| results.push(value.is_some()))?;
        Ok(results)
    }

    /// Visits ordered keys with borrowed value slices.
    pub fn visit_many<'a, I, F>(&mut self, keys: I, visitor: F) -> Result<()>
    where
        I: IntoIterator<Item = &'a [u8]>,
        F: FnMut(usize, Option<&[u8]>),
    {
        let keys = collect_and_validate_lookup_keys(keys, self.store.inner.options.key_len)?;
        self.process_many_slice(&keys, visitor)
    }

    /// Visits ordered keys already stored in a slice.
    pub fn visit_many_slice<K, F>(&mut self, keys: &[K], visitor: F) -> Result<()>
    where
        K: AsRef<[u8]>,
        F: FnMut(usize, Option<&[u8]>),
    {
        validate_lookup_key_slice(keys, self.store.inner.options.key_len)?;
        self.process_many_slice(keys, visitor)
    }

    fn process_many_slice<K, F>(&mut self, keys: &[K], mut visitor: F) -> Result<()>
    where
        K: AsRef<[u8]>,
        F: FnMut(usize, Option<&[u8]>),
    {
        let shard_policy = ShardPolicy::new(
            self.store.inner.options.shard_count,
            self.store.inner.options.shard_key_offset,
        );
        let options = LookupReadOptions {
            key_len: self.store.inner.options.key_len,
            value_layout: self.store.inner.options.value_layout,
            verify_block_checksums: self.store.inner.options.verify_block_checksums,
        };
        let mut start = 0usize;
        while start < keys.len() {
            let shard_id = shard_policy.shard_for_key(keys[start].as_ref());
            let mut end = start + 1;
            while end < keys.len() && shard_policy.shard_for_key(keys[end].as_ref()) == shard_id {
                end += 1;
            }

            let segments = {
                let state = self.store.inner.state.read();
                state.shards[shard_id].segments.clone()
            };
            process_shard_keys(
                segments.as_ref(),
                &mut self.shard_states[shard_id],
                &keys[start..end],
                start,
                options,
                &mut visitor,
            )?;
            start = end;
        }

        Ok(())
    }
}

pub(crate) fn collect_and_validate_lookup_keys<'a, I>(
    keys: I,
    key_len: usize,
) -> Result<Vec<&'a [u8]>>
where
    I: IntoIterator<Item = &'a [u8]>,
{
    let keys: Vec<_> = keys.into_iter().collect();
    validate_lookup_key_slice(&keys, key_len)?;
    Ok(keys)
}

pub(crate) fn validate_lookup_key_slice<K>(keys: &[K], key_len: usize) -> Result<()>
where
    K: AsRef<[u8]>,
{
    for key in keys {
        let key = key.as_ref();
        if key.len() != key_len {
            return Err(Error::WrongKeyLength {
                expected: key_len,
                actual: key.len(),
            });
        }
    }
    if !keys
        .windows(2)
        .all(|window| window[0].as_ref() <= window[1].as_ref())
    {
        return Err(Error::UnsortedLookupKeys);
    }
    Ok(())
}

pub(crate) fn fetch_from_shard(
    shard: &ShardState,
    key: &[u8],
    key_len: usize,
    value_layout: ValueLayout,
    verify_block_checksums: bool,
) -> Result<Option<Vec<u8>>> {
    let segment_index = match shard
        .segments
        .partition_point(|segment| segment.max_key.as_slice() < key)
    {
        idx if idx < shard.segments.len()
            && shard.segments[idx].min_key.as_slice() <= key
            && key <= shard.segments[idx].max_key.as_slice() =>
        {
            idx
        }
        _ => return Ok(None),
    };
    let segment = &shard.segments[segment_index];
    let block_index = segment.find_block_index(key);
    let block = match segment.load_block(block_index, key_len, value_layout, verify_block_checksums)
    {
        Ok(block) => block,
        Err(error) if error.is_cache_miss_corruption() => return Ok(None),
        Err(error) => return Err(error),
    };
    Ok(block.find_value(key))
}

fn process_shard_keys<K, F>(
    segments: &[Arc<SegmentState>],
    lookup_state: &mut ShardLookupState,
    keys: &[K],
    base_index: usize,
    options: LookupReadOptions,
    visitor: &mut F,
) -> Result<()>
where
    K: AsRef<[u8]>,
    F: FnMut(usize, Option<&[u8]>),
{
    if keys.is_empty() {
        return Ok(());
    }

    let mut key_index = 0usize;
    let mut segment_index = initial_segment_index(segments, lookup_state, keys[0].as_ref());
    reset_segment_if_needed(lookup_state, segment_index);

    while key_index < keys.len() {
        let key = keys[key_index].as_ref();

        if segment_index >= segments.len() {
            while key_index < keys.len() {
                visitor(base_index + key_index, None);
                key_index += 1;
            }
            break;
        }

        let segment = segments[segment_index].as_ref();
        if key > segment.max_key.as_slice() {
            segment_index += 1;
            reset_segment_if_needed(lookup_state, segment_index);
            continue;
        }

        if key < segment.min_key.as_slice() {
            while key_index < keys.len() && keys[key_index].as_ref() < segment.min_key.as_slice() {
                visitor(base_index + key_index, None);
                key_index += 1;
            }
            continue;
        }

        let block_index = initial_block_index(segment, lookup_state, key);
        lookup_state.current_segment_index = Some(segment_index);

        let block = match ensure_loaded_block(segment, lookup_state, block_index, options) {
            Ok(block) => block,
            Err(error) if error.is_cache_miss_corruption() => {
                let block_end = block_end_from_index(segment, block_index, keys, key_index);
                for offset in key_index..block_end {
                    visitor(base_index + offset, None);
                }
                key_index = block_end;
                lookup_state.current_block_index = block_index.checked_add(1);
                lookup_state.loaded_block = None;
                continue;
            }
            Err(error) => return Err(error),
        };
        let block_upper = block.last_key().min(segment.max_key.as_slice());
        let mut block_end = key_index;
        while block_end < keys.len() && keys[block_end].as_ref() <= block_upper {
            block_end += 1;
        }
        if block_end == key_index {
            let next_first_key = segment
                .block_index
                .get(block_index + 1)
                .map(|entry| entry.first_key.as_slice());
            if next_first_key.is_some_and(|next| key >= next) {
                lookup_state.current_block_index = block_index.checked_add(1);
                lookup_state.loaded_block = None;
            } else {
                visitor(base_index + key_index, None);
                key_index += 1;
            }
            continue;
        }

        process_block_keys(
            block,
            &keys[key_index..block_end],
            base_index + key_index,
            visitor,
        );
        key_index = block_end;
    }

    Ok(())
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

fn process_block_keys<K, F>(block: &DecodedBlock, keys: &[K], base_index: usize, visitor: &mut F)
where
    K: AsRef<[u8]>,
    F: FnMut(usize, Option<&[u8]>),
{
    let start_index = keys
        .first()
        .map_or(0, |key| block.lower_bound_index(key.as_ref()));
    let mut records = block.records_from(start_index).peekable();

    for (offset, key) in keys.iter().enumerate() {
        let key = key.as_ref();
        while let Some(record) = records.peek() {
            if record.key < key {
                let _ = records.next();
            } else {
                break;
            }
        }

        let value = records
            .peek()
            .filter(|record| record.key == key)
            .map(|record| record.value);
        visitor(base_index + offset, value);
    }
}

fn initial_segment_index(
    segments: &[Arc<SegmentState>],
    lookup_state: &ShardLookupState,
    key: &[u8],
) -> usize {
    let candidate = lookup_state
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

fn initial_block_index(
    segment: &SegmentState,
    lookup_state: &ShardLookupState,
    key: &[u8],
) -> usize {
    if let Some(block) = lookup_state.loaded_block.as_ref()
        && block.first_key() <= key
        && key <= block.last_key()
        && let Some(index) = lookup_state.current_block_index
    {
        return index;
    }

    if let Some(index) = lookup_state
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

fn reset_segment_if_needed(lookup_state: &mut ShardLookupState, segment_index: usize) {
    if lookup_state.current_segment_index != Some(segment_index) {
        lookup_state.current_segment_index = Some(segment_index);
        lookup_state.current_block_index = None;
        lookup_state.loaded_block = None;
    }
}

fn ensure_loaded_block<'a>(
    segment: &'a SegmentState,
    lookup_state: &'a mut ShardLookupState,
    block_index: usize,
    options: LookupReadOptions,
) -> Result<&'a DecodedBlock> {
    let needs_reload = lookup_state
        .loaded_block
        .as_ref()
        .is_none_or(|_| lookup_state.current_block_index != Some(block_index));
    if needs_reload {
        let buffer = lookup_state
            .loaded_block
            .take()
            .map(DecodedBlock::into_bytes)
            .unwrap_or_default();
        lookup_state.loaded_block = Some(segment.load_block_reusing(
            block_index,
            options.key_len,
            options.value_layout,
            options.verify_block_checksums,
            buffer,
        )?);
        lookup_state.current_block_index = Some(block_index);
    }

    Ok(lookup_state
        .loaded_block
        .as_ref()
        .expect("loaded block should exist"))
}
