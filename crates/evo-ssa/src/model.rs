use std::hash::Hash;

use rand::Rng;

/// A model with fixed reaction families and a dynamic set of concrete channels.
///
/// `ChannelKey` identifies one active concrete channel, such as `birth(species_i)` or
/// `mutation(species_i -> species_j)`. The scheduler owns the active channel table; the model owns
/// the biological semantics and tells the scheduler which concrete channels changed after an
/// event.
pub trait EvolvingModel {
    type State;
    type ChannelKey: Copy + Eq + Hash;
    type ChannelPayload: Clone;
    type Event;

    /// Build the initial simulation state.
    fn initial_state(&self) -> Self::State;

    /// Emit all initially active concrete channels.
    fn initialize_channels(
        &self,
        state: &Self::State,
        channels: &mut ChannelEditor<Self::ChannelKey, Self::ChannelPayload>,
    );

    /// Return the current propensity of one concrete channel.
    fn propensity(
        &self,
        state: &Self::State,
        key: Self::ChannelKey,
        payload: &Self::ChannelPayload,
        time: f64,
    ) -> f64;

    /// Fire one concrete channel and mutate the model state.
    fn fire<R: Rng + ?Sized>(
        &self,
        state: &mut Self::State,
        key: Self::ChannelKey,
        payload: &Self::ChannelPayload,
        rng: &mut R,
    ) -> Self::Event;

    /// Emit the concrete channel updates caused by `event`.
    fn refresh_after_event(
        &self,
        state: &Self::State,
        event: &Self::Event,
        channels: &mut ChannelEditor<Self::ChannelKey, Self::ChannelPayload>,
    );
}

/// A scheduler-facing update for one concrete channel.
#[derive(Debug, Clone, PartialEq)]
pub enum ChannelUpdate<K, P> {
    /// Insert a channel or replace the payload of an existing channel, then recompute propensity.
    Upsert { key: K, payload: P },
    /// Recompute propensity for an existing channel, keeping its payload.
    Recompute { key: K },
    /// Remove a channel from the active set.
    Remove { key: K },
}

/// Collects channel updates without exposing scheduler internals to the model.
#[derive(Debug, Clone)]
pub struct ChannelEditor<K, P> {
    updates: Vec<ChannelUpdate<K, P>>,
}

impl<K, P> Default for ChannelEditor<K, P> {
    fn default() -> Self {
        Self {
            updates: Vec::new(),
        }
    }
}

impl<K, P> ChannelEditor<K, P> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn upsert(&mut self, key: K, payload: P) {
        self.updates.push(ChannelUpdate::Upsert { key, payload });
    }

    pub fn recompute(&mut self, key: K) {
        self.updates.push(ChannelUpdate::Recompute { key });
    }

    pub fn remove(&mut self, key: K) {
        self.updates.push(ChannelUpdate::Remove { key });
    }

    pub fn is_empty(&self) -> bool {
        self.updates.is_empty()
    }

    pub fn len(&self) -> usize {
        self.updates.len()
    }

    pub fn drain(&mut self) -> impl Iterator<Item = ChannelUpdate<K, P>> + '_ {
        self.updates.drain(..)
    }

    pub fn into_updates(self) -> Vec<ChannelUpdate<K, P>> {
        self.updates
    }
}
