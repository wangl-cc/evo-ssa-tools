use std::{collections::HashMap, hash::Hash};

use rand::{Rng, RngExt};

use crate::{
    Error, Result,
    engine::FamilySimulation,
    model::{ChannelUpdate, EvolvingModel},
    scheduler::{
        FamilyAlgorithm, FamilyUpdateContext, ScheduledEvent, ScheduledFamilyChannel, SsaScheduler,
        family::{FamilyId, FamilyReactionSet, StaticFamilyModel},
        math::{checked_propensity, draw_exponential},
        update::{FamilyChannelUpdates, FamilyUpdateBuffer},
        weighted::SegmentTree,
    },
};

/// Cached Direct Method scheduler for arbitrary concrete channel keys.
///
/// The scheduler stores one propensity per active channel and samples from the cached total using a
/// segment tree. Updating one channel is `O(log M)`, where `M` is the number of allocated channel
/// slots. The hash table maps model-owned channel keys to stable scheduler slots.
#[derive(Debug, Clone)]
pub struct DirectScheduler<K, P> {
    slots: Vec<Option<ChannelSlot<K, P>>>,
    slot_by_key: HashMap<K, usize>,
    free_slots: Vec<usize>,
    weights: SegmentTree,
    active_channel_count: usize,
}

impl<K, P> Default for DirectScheduler<K, P> {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            slot_by_key: HashMap::new(),
            free_slots: Vec::new(),
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
    /// Create an empty scheduler. [`Simulation`](crate::Simulation) initializes its channels.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return current channel-table and propensity-cache statistics.
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

        let slot_index = if let Some(slot_index) = self.free_slots.pop() {
            self.slots[slot_index] = Some(ChannelSlot { key, payload });
            self.weights.set(slot_index, propensity);
            slot_index
        } else {
            let slot_index = self.slots.len();
            self.slots.push(Some(ChannelSlot { key, payload }));
            self.weights.push(propensity);
            slot_index
        };
        self.slot_by_key.insert(key, slot_index);
        self.active_channel_count += 1;
        Ok(())
    }

    fn recompute<M>(&mut self, model: &M, state: &M::State, time: f64, key: K) -> Result<()>
    where
        M: EvolvingModel<ChannelKey = K, ChannelPayload = P>,
    {
        let Some(&slot_index) = self.slot_by_key.get(&key) else {
            return Err(Error::MissingRecomputeChannel);
        };
        let slot = self.slots[slot_index]
            .as_ref()
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
            self.free_slots.push(slot_index);
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

    fn next_event<C, S>(
        &mut self,
        clock_rng: &mut C,
        selection_rng: &mut S,
    ) -> Result<Option<ScheduledEvent<K, P>>>
    where
        C: Rng + ?Sized,
        S: Rng + ?Sized,
    {
        let total_propensity = self.weights.total();
        if total_propensity <= 0.0 {
            return Ok(None);
        }

        let dt = draw_exponential(total_propensity, clock_rng);
        let reaction_draw = selection_rng.random::<f64>() * total_propensity;
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

/// Two-level cached Direct Method algorithm for compile-time reaction families.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct FamilyDirectAlgorithm {
    samplers: Vec<SegmentTree>,
}

/// Two-level cached Direct Method simulation for compile-time reaction families.
///
/// Each family owns a segment tree over its local channels. Sampling first chooses a family from
/// its cached total, then chooses a local channel within that family.
pub type FamilyDirect<M, F> = FamilySimulation<M, F, FamilyDirectAlgorithm, FamilyChannelUpdates>;

impl<M, F> FamilySimulation<M, F, FamilyDirectAlgorithm, FamilyChannelUpdates>
where
    M: StaticFamilyModel<Families = F>,
    F: FamilyReactionSet<M>,
{
    /// Initialize every concrete channel at time zero.
    pub fn new(model: M, families: F) -> Result<Self> {
        let state = model.initial_state();
        let algorithm = FamilyDirectAlgorithm::new(&model, &state, &families)?;
        Ok(Self::from_initialized(
            model,
            state,
            families,
            algorithm,
            FamilyChannelUpdates::default(),
        ))
    }
}

impl FamilyDirectAlgorithm {
    fn new<M, F>(model: &M, state: &M::State, families: &F) -> Result<Self>
    where
        M: StaticFamilyModel<Families = F>,
        F: FamilyReactionSet<M>,
    {
        let mut samplers = Vec::with_capacity(F::FAMILY_COUNT);
        for family in 0..F::FAMILY_COUNT {
            let mut sampler = SegmentTree::default();
            let channel_count =
                families
                    .channel_count(model, state, family)
                    .ok_or(Error::InvalidFamily {
                        family,
                        family_count: F::FAMILY_COUNT,
                    })?;
            sampler.resize_with(channel_count, |local_channel| {
                let propensity = families
                    .propensity(model, state, family, local_channel, 0.0)
                    .ok_or(Error::InvalidFamily {
                        family,
                        family_count: F::FAMILY_COUNT,
                    })?;
                checked_propensity(propensity)
            })?;
            samplers.push(sampler);
        }
        Ok(Self { samplers })
    }

    fn resize_samplers<M, F>(
        &mut self,
        model: &M,
        state: &M::State,
        families: &F,
        time: f64,
    ) -> Result<()>
    where
        M: StaticFamilyModel<Families = F>,
        F: FamilyReactionSet<M>,
    {
        for family in 0..F::FAMILY_COUNT {
            let channel_count =
                families
                    .channel_count(model, state, family)
                    .ok_or(Error::InvalidFamily {
                        family,
                        family_count: F::FAMILY_COUNT,
                    })?;
            self.samplers[family].resize_with(channel_count, |local_channel| {
                let propensity = families
                    .propensity(model, state, family, local_channel, time)
                    .ok_or(Error::InvalidFamily {
                        family,
                        family_count: F::FAMILY_COUNT,
                    })?;
                checked_propensity(propensity)
            })?;
        }
        Ok(())
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

impl<M, F> FamilyAlgorithm<M, F> for FamilyDirectAlgorithm
where
    M: StaticFamilyModel<Families = F>,
    F: FamilyReactionSet<M>,
{
    fn next_event<C, S>(
        &mut self,
        now: f64,
        clock_rng: &mut C,
        selection_rng: &mut S,
    ) -> Result<Option<ScheduledFamilyChannel<F>>>
    where
        C: Rng + ?Sized,
        S: Rng + ?Sized,
    {
        let total_propensity = self.total_propensity();
        if total_propensity <= 0.0 {
            return Ok(None);
        }
        let (family, local_channel) =
            self.sample_channel(selection_rng.random::<f64>() * total_propensity)?;
        Ok(Some(ScheduledFamilyChannel {
            // SAFETY: `sample_channel` returns an index from the configured family samplers.
            family: unsafe { FamilyId::__from_index_unchecked(family) },
            local_channel,
            time: now + draw_exponential(total_propensity, clock_rng),
        }))
    }

    fn apply_updates<C, U>(
        &mut self,
        context: FamilyUpdateContext<'_, M, F, C>,
        updates: &mut U,
    ) -> Result<()>
    where
        C: Rng + ?Sized,
        U: FamilyUpdateBuffer,
    {
        self.resize_samplers(context.model, context.state, context.families, context.time)?;

        while let Some(update) = updates.pop() {
            let Some(sampler) = self.samplers.get_mut(update.family) else {
                return Err(Error::InvalidFamily {
                    family: update.family,
                    family_count: F::FAMILY_COUNT,
                });
            };
            if update.local_channel >= sampler.len() {
                return Err(Error::InvalidFamilyChannel {
                    family: update.family,
                    local_channel: update.local_channel,
                    channel_count: sampler.len(),
                });
            }
            let propensity = context
                .families
                .propensity(
                    context.model,
                    context.state,
                    update.family,
                    update.local_channel,
                    context.time,
                )
                .ok_or(Error::InvalidFamily {
                    family: update.family,
                    family_count: F::FAMILY_COUNT,
                })?;
            sampler.set(update.local_channel, checked_propensity(propensity)?);
        }
        Ok(())
    }
}

/// Snapshot of a [`DirectScheduler`]'s channel table and cached total.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirectSchedulerStats {
    /// Number of channels currently eligible for updates and sampling.
    pub active_channel_count: usize,
    /// Number of reusable scheduler slots allocated at the peak active-channel count.
    pub allocated_slot_count: usize,
    /// Sum of all cached active-channel propensities.
    pub total_propensity: f64,
}

#[derive(Debug, Clone)]
struct ChannelSlot<K, P> {
    key: K,
    payload: P,
}
