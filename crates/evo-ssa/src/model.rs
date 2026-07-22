use std::hash::Hash;

use rand::Rng;

/// A model with fixed reaction families and a dynamic set of concrete channels.
///
/// `ChannelKey` identifies one active concrete channel, such as `birth(species_i)` or
/// `mutation(species_i -> species_j)`. The scheduler owns the active channel table; the model owns
/// the biological semantics and tells the scheduler which concrete channels changed after an
/// event.
pub trait EvolvingModel {
    /// Mutable biological state advanced by reaction events.
    type State;
    /// Stable identity of one active concrete reaction channel.
    type ChannelKey: Copy + Eq + Hash;
    /// Scheduler-owned data needed to evaluate or fire a channel.
    type ChannelPayload: Clone;
    /// Compact description of one fired reaction and its affected state.
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
    ///
    /// Schedulers cache this value until the model emits a matching update from
    /// [`refresh_after_event`](Self::refresh_after_event). Propensities must therefore be
    /// piecewise-constant between emitted updates; time-dependent hazards need to emit updates at
    /// every point where cached propensities change.
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
    Upsert {
        /// Stable channel identity.
        key: K,
        /// Payload used for propensity evaluation and firing.
        payload: P,
    },
    /// Recompute propensity for an existing channel, keeping its payload.
    Recompute {
        /// Existing channel to recompute.
        key: K,
    },
    /// Remove a channel from the active set.
    Remove {
        /// Existing channel to remove, if present.
        key: K,
    },
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
    /// Create an empty update collector.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a channel or replace its payload, then recompute its propensity.
    pub fn upsert(&mut self, key: K, payload: P) {
        self.updates.push(ChannelUpdate::Upsert { key, payload });
    }

    /// Request propensity recomputation for an existing channel.
    pub fn recompute(&mut self, key: K) {
        self.updates.push(ChannelUpdate::Recompute { key });
    }

    /// Remove a channel if it is active.
    pub fn remove(&mut self, key: K) {
        self.updates.push(ChannelUpdate::Remove { key });
    }

    /// Return whether no updates have been collected.
    pub fn is_empty(&self) -> bool {
        self.updates.is_empty()
    }

    /// Return the number of collected updates.
    pub fn len(&self) -> usize {
        self.updates.len()
    }

    /// Drain collected updates in insertion order while retaining allocated capacity.
    pub fn drain(&mut self) -> impl Iterator<Item = ChannelUpdate<K, P>> + '_ {
        self.updates.drain(..)
    }

    /// Consume the editor and return its updates in insertion order.
    pub fn into_updates(self) -> Vec<ChannelUpdate<K, P>> {
        self.updates
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn channel_editor_preserves_update_order() {
        let mut editor = ChannelEditor::new();
        editor.upsert(1, "birth");
        editor.recompute(2);
        editor.remove(3);

        assert_eq!(editor.len(), 3);
        assert!(!editor.is_empty());
        assert_eq!(editor.into_updates(), vec![
            ChannelUpdate::Upsert {
                key: 1,
                payload: "birth"
            },
            ChannelUpdate::Recompute { key: 2 },
            ChannelUpdate::Remove { key: 3 },
        ]);
    }

    #[test]
    fn channel_editor_drain_clears_buffer() {
        let mut editor: ChannelEditor<usize, ()> = ChannelEditor::new();
        editor.recompute(1);
        editor.recompute(2);

        let updates: Vec<_> = editor.drain().collect();

        assert_eq!(updates, vec![
            ChannelUpdate::Recompute { key: 1 },
            ChannelUpdate::Recompute { key: 2 },
        ]);
        assert!(editor.is_empty());
    }
}
