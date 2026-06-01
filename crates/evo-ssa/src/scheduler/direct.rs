use std::{collections::HashMap, hash::Hash};

use rand::{Rng, RngExt};

use crate::{
    Error, Result,
    model::{ChannelUpdate, EvolvingModel},
    scheduler::{
        ScheduledEvent, SsaScheduler,
        update::{
            FamilyChannelUpdate, FamilyChannelUpdates, StaticChannelUpdates, StaticUpdateSink,
        },
    },
};

/// Cached Direct Method scheduler for a dynamic set of concrete channels.
///
/// The scheduler stores one cached propensity per active channel and samples from their total using
/// a segment tree. Updating one channel is `O(log M)`, where `M` is the number of allocated channel
/// slots.
#[derive(Debug, Clone)]
pub struct DirectScheduler<K, P> {
    slots: Vec<Option<ChannelSlot<K, P>>>,
    slot_by_key: HashMap<K, usize>,
    weights: SegmentTree,
    active_channel_count: usize,
}

impl<K, P> Default for DirectScheduler<K, P> {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            slot_by_key: HashMap::new(),
            weights: SegmentTree::default(),
            active_channel_count: 0,
        }
    }
}

impl<K, P> DirectScheduler<K, P>
where
    K: Copy + Eq + Hash,
    P: Clone,
{
    pub fn new() -> Self {
        Self::default()
    }

    pub fn stats(&self) -> DirectSchedulerStats {
        DirectSchedulerStats {
            active_channel_count: self.active_channel_count,
            allocated_slot_count: self.slots.len(),
            total_propensity: self.weights.total(),
        }
    }

    fn upsert<M>(
        &mut self,
        model: &M,
        state: &M::State,
        time: f64,
        key: K,
        payload: P,
    ) -> Result<()>
    where
        M: EvolvingModel<ChannelKey = K, ChannelPayload = P>,
    {
        let propensity = checked_propensity(model.propensity(state, key, &payload, time))?;

        if let Some(&slot_index) = self.slot_by_key.get(&key) {
            self.slots[slot_index] = Some(ChannelSlot { key, payload });
            self.weights.set(slot_index, propensity);
            return Ok(());
        }

        let slot_index = self.slots.len();
        self.slots.push(Some(ChannelSlot { key, payload }));
        self.slot_by_key.insert(key, slot_index);
        self.weights.push(propensity);
        self.active_channel_count += 1;
        Ok(())
    }

    fn recompute<M>(&mut self, model: &M, state: &M::State, time: f64, key: K) -> Result<()>
    where
        M: EvolvingModel<ChannelKey = K, ChannelPayload = P>,
    {
        let Some(&slot_index) = self.slot_by_key.get(&key) else {
            return Ok(());
        };
        let slot = self.slots[slot_index]
            .as_mut()
            .ok_or(Error::MissingSelectedChannel)?;
        let propensity = checked_propensity(model.propensity(state, key, &slot.payload, time))?;
        self.weights.set(slot_index, propensity);
        Ok(())
    }

    fn remove(&mut self, key: K) {
        let Some(slot_index) = self.slot_by_key.remove(&key) else {
            return;
        };
        if self.slots[slot_index].take().is_some() {
            self.weights.set(slot_index, 0.0);
            self.active_channel_count -= 1;
        }
    }
}

impl<M, K, P> SsaScheduler<M> for DirectScheduler<K, P>
where
    M: EvolvingModel<ChannelKey = K, ChannelPayload = P>,
    K: Copy + Eq + Hash,
    P: Clone,
{
    fn initialize(
        &mut self,
        model: &M,
        state: &M::State,
        time: f64,
        channels: impl IntoIterator<Item = ChannelUpdate<K, P>>,
    ) -> Result<()> {
        *self = Self::default();
        self.apply_updates(model, state, time, channels)
    }

    fn next_event<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<Option<ScheduledEvent<K, P>>> {
        let total_propensity = self.weights.total();
        if total_propensity <= 0.0 {
            return Ok(None);
        }

        let waiting_time_draw = rng.random::<f64>().clamp(f64::MIN_POSITIVE, 1.0);
        let dt = -waiting_time_draw.ln() / total_propensity;
        let reaction_draw = rng.random::<f64>() * total_propensity;
        let slot_index = self
            .weights
            .sample(reaction_draw)
            .ok_or(Error::MissingSelectedChannel)?;
        let slot = self.slots[slot_index]
            .as_ref()
            .ok_or(Error::MissingSelectedChannel)?;

        Ok(Some(ScheduledEvent {
            key: slot.key,
            payload: slot.payload.clone(),
            dt,
        }))
    }

    fn apply_updates(
        &mut self,
        model: &M,
        state: &M::State,
        time: f64,
        updates: impl IntoIterator<Item = ChannelUpdate<K, P>>,
    ) -> Result<()> {
        for update in updates {
            match update {
                ChannelUpdate::Upsert { key, payload } => {
                    self.upsert(model, state, time, key, payload)?;
                }
                ChannelUpdate::Recompute { key } => {
                    self.recompute(model, state, time, key)?;
                }
                ChannelUpdate::Remove { key } => self.remove(key),
            }
        }
        Ok(())
    }

    fn active_channel_count(&self) -> usize {
        self.active_channel_count
    }
}

/// Cached Direct Method scheduler for dense `usize` channel ids.
///
/// This scheduler is intended for models whose fixed reaction families can encode concrete
/// channels directly, for example `birth(i) = 2 * i` and `death(i) = 2 * i + 1`. It avoids the
/// `HashMap<ChannelKey, slot>` lookup used by [`DirectScheduler`].
#[derive(Debug, Clone)]
pub struct DenseDirectScheduler<P> {
    slots: Vec<Option<ChannelSlot<usize, P>>>,
    weights: SegmentTree,
    active_channel_count: usize,
}

impl<P> Default for DenseDirectScheduler<P> {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            weights: SegmentTree::default(),
            active_channel_count: 0,
        }
    }
}

impl<P> DenseDirectScheduler<P>
where
    P: Clone,
{
    pub fn new() -> Self {
        Self::default()
    }

    pub fn stats(&self) -> DirectSchedulerStats {
        DirectSchedulerStats {
            active_channel_count: self.active_channel_count,
            allocated_slot_count: self.slots.len(),
            total_propensity: self.weights.total(),
        }
    }

    fn upsert<M>(
        &mut self,
        model: &M,
        state: &M::State,
        time: f64,
        key: usize,
        payload: P,
    ) -> Result<()>
    where
        M: EvolvingModel<ChannelKey = usize, ChannelPayload = P>,
    {
        let propensity = checked_propensity(model.propensity(state, key, &payload, time))?;

        if key >= self.slots.len() {
            self.slots.resize_with(key + 1, || None);
        }

        if self.slots[key].is_none() {
            self.active_channel_count += 1;
        }
        self.slots[key] = Some(ChannelSlot { key, payload });
        self.weights.set(key, propensity);
        Ok(())
    }

    fn recompute<M>(&mut self, model: &M, state: &M::State, time: f64, key: usize) -> Result<()>
    where
        M: EvolvingModel<ChannelKey = usize, ChannelPayload = P>,
    {
        let Some(Some(slot)) = self.slots.get(key) else {
            return Ok(());
        };
        let propensity = checked_propensity(model.propensity(state, key, &slot.payload, time))?;
        self.weights.set(key, propensity);
        Ok(())
    }

    fn remove(&mut self, key: usize) {
        let Some(slot) = self.slots.get_mut(key) else {
            return;
        };
        if slot.take().is_some() {
            self.weights.set(key, 0.0);
            self.active_channel_count -= 1;
        }
    }
}

impl<M, P> SsaScheduler<M> for DenseDirectScheduler<P>
where
    M: EvolvingModel<ChannelKey = usize, ChannelPayload = P>,
    P: Clone,
{
    fn initialize(
        &mut self,
        model: &M,
        state: &M::State,
        time: f64,
        channels: impl IntoIterator<Item = ChannelUpdate<usize, P>>,
    ) -> Result<()> {
        *self = Self::default();
        self.apply_updates(model, state, time, channels)
    }

    fn next_event<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
    ) -> Result<Option<ScheduledEvent<usize, P>>> {
        let total_propensity = self.weights.total();
        if total_propensity <= 0.0 {
            return Ok(None);
        }

        let waiting_time_draw = rng.random::<f64>().clamp(f64::MIN_POSITIVE, 1.0);
        let dt = -waiting_time_draw.ln() / total_propensity;
        let reaction_draw = rng.random::<f64>() * total_propensity;
        let slot_index = self
            .weights
            .sample(reaction_draw)
            .ok_or(Error::MissingSelectedChannel)?;
        let slot = self.slots[slot_index]
            .as_ref()
            .ok_or(Error::MissingSelectedChannel)?;

        Ok(Some(ScheduledEvent {
            key: slot.key,
            payload: slot.payload.clone(),
            dt,
        }))
    }

    fn apply_updates(
        &mut self,
        model: &M,
        state: &M::State,
        time: f64,
        updates: impl IntoIterator<Item = ChannelUpdate<usize, P>>,
    ) -> Result<()> {
        for update in updates {
            match update {
                ChannelUpdate::Upsert { key, payload } => {
                    self.upsert(model, state, time, key, payload)?;
                }
                ChannelUpdate::Recompute { key } => {
                    self.recompute(model, state, time, key)?;
                }
                ChannelUpdate::Remove { key } => self.remove(key),
            }
        }
        Ok(())
    }

    fn active_channel_count(&self) -> usize {
        self.active_channel_count
    }
}

/// Model contract for statically dispatched direct SSA.
pub trait StaticDirectModel {
    type State;

    fn initial_state(&self) -> Self::State;
}

/// One compile-time reaction family in a statically dispatched direct SSA model.
pub trait StaticReactionFamily<M: StaticDirectModel> {
    fn channel_count(&self, model: &M, state: &M::State) -> usize;

    fn propensity(&self, model: &M, state: &M::State, local_channel: usize, time: f64) -> f64;

    fn fire<R, U>(
        &self,
        model: &M,
        state: &mut M::State,
        local_channel: usize,
        rng: &mut R,
        updates: &mut U,
    ) where
        R: Rng + ?Sized,
        U: StaticUpdateSink + ?Sized;
}

/// A compile-time set of reaction families.
pub trait StaticReactionFamilySet<M: StaticDirectModel> {
    fn channel_count(&self, model: &M, state: &M::State) -> usize;

    fn propensity(&self, model: &M, state: &M::State, channel: usize, time: f64) -> f64;

    fn fire<R: Rng + ?Sized>(
        &self,
        model: &M,
        state: &mut M::State,
        channel: usize,
        rng: &mut R,
        updates: &mut Vec<usize>,
    );
}

/// A compile-time set of reaction families stored as separate propensity samplers.
pub trait FamilySeparatedReactionSet<M: StaticDirectModel> {
    fn family_count(&self) -> usize;

    fn channel_count(&self, model: &M, state: &M::State, family: usize) -> usize;

    fn propensity(
        &self,
        model: &M,
        state: &M::State,
        family: usize,
        local_channel: usize,
        time: f64,
    ) -> f64;

    fn fire<R: Rng + ?Sized>(
        &self,
        model: &M,
        state: &mut M::State,
        selected: FamilyChannelUpdate,
        rng: &mut R,
        updates: &mut FamilyChannelUpdates<'_>,
    );
}

macro_rules! impl_reaction_family_sets {
    ($count:expr; $( $idx:tt => $ty:ident ),+ $(,)?) => {
        impl<M, $( $ty ),+> StaticReactionFamilySet<M> for ($( $ty, )+)
        where
            M: StaticDirectModel,
            $( $ty: StaticReactionFamily<M>, )+
        {
            fn channel_count(&self, model: &M, state: &M::State) -> usize {
                0 $(+ self.$idx.channel_count(model, state))+
            }

            fn propensity(&self, model: &M, state: &M::State, channel: usize, time: f64) -> f64 {
                let mut offset = 0;
                $(
                    let count = self.$idx.channel_count(model, state);
                    if channel < offset + count {
                        return self.$idx.propensity(model, state, channel - offset, time);
                    }
                    offset += count;
                )+
                let _ = offset;
                0.0
            }

            fn fire<R: Rng + ?Sized>(
                &self,
                model: &M,
                state: &mut M::State,
                channel: usize,
                rng: &mut R,
                updates: &mut Vec<usize>,
            ) {
                let mut offset = 0;
                let family_offsets = [$({
                    let current = offset;
                    offset += self.$idx.channel_count(model, state);
                    current
                }),+];
                let _ = offset;

                let mut selected_offset = 0;
                $(
                    let count = self.$idx.channel_count(model, state);
                    if channel < selected_offset + count {
                        self.$idx.fire(
                            model,
                            state,
                            channel - selected_offset,
                            rng,
                            &mut StaticChannelUpdates::global(
                                $idx,
                                selected_offset,
                                &family_offsets,
                                updates,
                            ),
                        );
                        return;
                    }
                    selected_offset += count;
                )+
                let _ = selected_offset;
            }
        }

        impl<M, $( $ty ),+> FamilySeparatedReactionSet<M> for ($( $ty, )+)
        where
            M: StaticDirectModel,
            $( $ty: StaticReactionFamily<M>, )+
        {
            fn family_count(&self) -> usize {
                $count
            }

            fn channel_count(&self, model: &M, state: &M::State, family: usize) -> usize {
                match family {
                    $(
                        $idx => self.$idx.channel_count(model, state),
                    )+
                    _ => 0,
                }
            }

            fn propensity(
                &self,
                model: &M,
                state: &M::State,
                family: usize,
                local_channel: usize,
                time: f64,
            ) -> f64 {
                match family {
                    $(
                        $idx => self.$idx.propensity(model, state, local_channel, time),
                    )+
                    _ => 0.0,
                }
            }

            fn fire<R: Rng + ?Sized>(
                &self,
                model: &M,
                state: &mut M::State,
                selected: FamilyChannelUpdate,
                rng: &mut R,
                updates: &mut FamilyChannelUpdates<'_>,
            ) {
                let Some(offset) = updates.offset_for(selected.family) else {
                    return;
                };
                match selected.family {
                    $(
                        $idx => self.$idx.fire(
                            model,
                            state,
                            selected.local_channel,
                            rng,
                            &mut updates.writer(selected.family, offset),
                        ),
                    )+
                    _ => {}
                }
            }
        }
    };
}

impl_reaction_family_sets!(1; 0 => A);
impl_reaction_family_sets!(2; 0 => A, 1 => B);
impl_reaction_family_sets!(3; 0 => A, 1 => B, 2 => C);
impl_reaction_family_sets!(4; 0 => A, 1 => B, 2 => C, 3 => D);
impl_reaction_family_sets!(5; 0 => A, 1 => B, 2 => C, 3 => D, 4 => E);
impl_reaction_family_sets!(6; 0 => A, 1 => B, 2 => C, 3 => D, 4 => E, 5 => G);

/// Cached Direct Method with compile-time reaction families and dense channel ids.
#[derive(Debug, Clone)]
pub struct StaticDirect<M, F>
where
    M: StaticDirectModel,
{
    model: M,
    state: M::State,
    families: F,
    weights: SegmentTree,
    updates: Vec<usize>,
    time: f64,
}

impl<M, F> StaticDirect<M, F>
where
    M: StaticDirectModel,
    F: StaticReactionFamilySet<M>,
{
    pub fn new(model: M, families: F) -> Result<Self> {
        let state = model.initial_state();
        let mut weights = SegmentTree::default();
        let time = 0.0;
        for channel in 0..families.channel_count(&model, &state) {
            let propensity =
                checked_propensity(families.propensity(&model, &state, channel, time))?;
            weights.push(propensity);
        }

        Ok(Self {
            model,
            state,
            families,
            weights,
            updates: Vec::new(),
            time,
        })
    }

    pub fn time(&self) -> f64 {
        self.time
    }

    pub fn state(&self) -> &M::State {
        &self.state
    }

    pub fn step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<bool> {
        let total_propensity = self.weights.total();
        if total_propensity <= 0.0 {
            return Ok(false);
        }

        let waiting_time_draw = rng.random::<f64>().clamp(f64::MIN_POSITIVE, 1.0);
        self.time += -waiting_time_draw.ln() / total_propensity;
        let reaction_draw = rng.random::<f64>() * total_propensity;
        let channel = self
            .weights
            .sample(reaction_draw)
            .ok_or(Error::MissingSelectedChannel)?;

        self.updates.clear();
        self.families.fire(
            &self.model,
            &mut self.state,
            channel,
            rng,
            &mut self.updates,
        );
        for channel in self.updates.drain(..) {
            let propensity = checked_propensity(self.families.propensity(
                &self.model,
                &self.state,
                channel,
                self.time,
            ))?;
            self.weights.set(channel, propensity);
        }

        Ok(true)
    }

    pub fn run<R: Rng + ?Sized>(&mut self, rng: &mut R, max_events: usize) -> Result<usize> {
        let mut fired_events = 0;
        while fired_events < max_events && self.step(rng)? {
            fired_events += 1;
        }
        Ok(fired_events)
    }
}

/// Two-level cached Direct Method with one propensity sampler per reaction family.
#[derive(Debug, Clone)]
pub struct FamilyDirect<M, F>
where
    M: StaticDirectModel,
{
    model: M,
    state: M::State,
    families: F,
    samplers: Vec<SegmentTree>,
    family_offsets: Vec<usize>,
    updates: Vec<FamilyChannelUpdate>,
    time: f64,
}

impl<M, F> FamilyDirect<M, F>
where
    M: StaticDirectModel,
    F: FamilySeparatedReactionSet<M>,
{
    pub fn new(model: M, families: F) -> Result<Self> {
        let state = model.initial_state();
        let time = 0.0;
        let mut samplers = Vec::with_capacity(families.family_count());
        let mut family_offsets = Vec::with_capacity(families.family_count());
        let mut offset = 0;

        for family in 0..families.family_count() {
            family_offsets.push(offset);
            let channel_count = families.channel_count(&model, &state, family);
            let mut sampler = SegmentTree::default();
            for local_channel in 0..channel_count {
                let propensity = checked_propensity(families.propensity(
                    &model,
                    &state,
                    family,
                    local_channel,
                    time,
                ))?;
                sampler.push(propensity);
            }
            offset += channel_count;
            samplers.push(sampler);
        }

        Ok(Self {
            model,
            state,
            families,
            samplers,
            family_offsets,
            updates: Vec::new(),
            time,
        })
    }

    pub fn time(&self) -> f64 {
        self.time
    }

    pub fn state(&self) -> &M::State {
        &self.state
    }

    pub fn step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<bool> {
        let total_propensity = self.total_propensity();
        if total_propensity <= 0.0 {
            return Ok(false);
        }

        let waiting_time_draw = rng.random::<f64>().clamp(f64::MIN_POSITIVE, 1.0);
        self.time += -waiting_time_draw.ln() / total_propensity;

        let (family, local_channel) =
            self.sample_channel(rng.random::<f64>() * total_propensity)?;
        self.updates.clear();
        let selected = FamilyChannelUpdate {
            family,
            local_channel,
        };
        let mut updates = FamilyChannelUpdates::new(&self.family_offsets, &mut self.updates);
        self.families
            .fire(&self.model, &mut self.state, selected, rng, &mut updates);

        for update in self.updates.drain(..) {
            if update.family >= self.samplers.len() {
                continue;
            }
            let propensity = checked_propensity(self.families.propensity(
                &self.model,
                &self.state,
                update.family,
                update.local_channel,
                self.time,
            ))?;
            self.samplers[update.family].set(update.local_channel, propensity);
        }

        Ok(true)
    }

    pub fn run<R: Rng + ?Sized>(&mut self, rng: &mut R, max_events: usize) -> Result<usize> {
        let mut fired_events = 0;
        while fired_events < max_events && self.step(rng)? {
            fired_events += 1;
        }
        Ok(fired_events)
    }

    fn total_propensity(&self) -> f64 {
        self.samplers.iter().map(SegmentTree::total).sum()
    }

    fn sample_channel(&self, target: f64) -> Result<(usize, usize)> {
        let mut target = target;
        for (family, sampler) in self.samplers.iter().enumerate() {
            let family_total = sampler.total();
            if target < family_total {
                let local_channel = sampler
                    .sample(target)
                    .ok_or(Error::MissingSelectedChannel)?;
                return Ok((family, local_channel));
            }
            target -= family_total;
        }
        Err(Error::MissingSelectedChannel)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirectSchedulerStats {
    pub active_channel_count: usize,
    pub allocated_slot_count: usize,
    pub total_propensity: f64,
}

#[derive(Debug, Clone)]
struct ChannelSlot<K, P> {
    key: K,
    payload: P,
}

fn checked_propensity(value: f64) -> Result<f64> {
    if value.is_finite() && value >= 0.0 {
        Ok(value)
    } else {
        Err(Error::InvalidPropensity { value })
    }
}

#[derive(Debug, Clone, Default)]
struct SegmentTree {
    weights: Vec<f64>,
    tree: Vec<f64>,
    leaf_count: usize,
}

impl SegmentTree {
    fn total(&self) -> f64 {
        self.tree.get(1).copied().unwrap_or(0.0)
    }

    fn push(&mut self, weight: f64) {
        self.weights.push(weight);
        self.ensure_capacity(self.weights.len());
        self.set(self.weights.len() - 1, weight);
    }

    fn set(&mut self, index: usize, weight: f64) {
        if index >= self.weights.len() {
            self.weights.resize(index + 1, 0.0);
        }
        self.ensure_capacity(index + 1);
        self.weights[index] = weight;

        let mut tree_index = self.leaf_count + index;
        self.tree[tree_index] = weight;
        while tree_index > 1 {
            tree_index /= 2;
            self.tree[tree_index] = self.tree[tree_index * 2] + self.tree[tree_index * 2 + 1];
        }
    }

    fn sample(&self, target: f64) -> Option<usize> {
        if target < 0.0 || target >= self.total() {
            return None;
        }

        let mut target = target;
        let mut tree_index = 1;
        while tree_index < self.leaf_count {
            let left = tree_index * 2;
            if target < self.tree[left] {
                tree_index = left;
            } else {
                target -= self.tree[left];
                tree_index = left + 1;
            }
        }

        let index = tree_index - self.leaf_count;
        (index < self.weights.len() && self.weights[index] > 0.0).then_some(index)
    }

    fn ensure_capacity(&mut self, len: usize) {
        if len <= self.leaf_count {
            return;
        }

        self.leaf_count = len.next_power_of_two().max(1);
        self.tree = vec![0.0; self.leaf_count * 2];
        for (index, &weight) in self.weights.iter().enumerate() {
            self.tree[self.leaf_count + index] = weight;
        }
        for index in (1..self.leaf_count).rev() {
            self.tree[index] = self.tree[index * 2] + self.tree[index * 2 + 1];
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn segment_tree_samples_weighted_slots() {
        let mut tree = SegmentTree::default();
        tree.push(0.0);
        tree.push(2.0);
        tree.push(3.0);

        assert_eq!(tree.total(), 5.0);
        assert_eq!(tree.sample(0.0), Some(1));
        assert_eq!(tree.sample(1.9), Some(1));
        assert_eq!(tree.sample(2.0), Some(2));
        assert_eq!(tree.sample(4.9), Some(2));
    }

    #[test]
    fn segment_tree_handles_growth_and_updates() {
        let mut tree = SegmentTree::default();
        for _ in 0..17 {
            tree.push(1.0);
        }
        assert_eq!(tree.total(), 17.0);

        tree.set(8, 0.0);
        assert_eq!(tree.total(), 16.0);
        assert_eq!(tree.sample(8.0), Some(9));
    }
}
