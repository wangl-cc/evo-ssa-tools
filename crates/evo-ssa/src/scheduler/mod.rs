pub mod direct;
pub mod nrm;
pub mod update;

use rand::Rng;

use crate::{
    Result,
    model::{ChannelUpdate, EvolvingModel},
};

/// A concrete channel selected by a scheduler.
#[derive(Debug, Clone)]
pub struct ScheduledEvent<K, P> {
    pub key: K,
    pub payload: P,
    pub dt: f64,
}

/// Common interface for exact SSA schedulers.
pub trait SsaScheduler<M: EvolvingModel> {
    fn initialize(
        &mut self,
        model: &M,
        state: &M::State,
        time: f64,
        channels: impl IntoIterator<Item = ChannelUpdate<M::ChannelKey, M::ChannelPayload>>,
    ) -> Result<()>;

    fn next_event<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
    ) -> Result<Option<ScheduledEvent<M::ChannelKey, M::ChannelPayload>>>;

    fn apply_updates(
        &mut self,
        model: &M,
        state: &M::State,
        time: f64,
        updates: impl IntoIterator<Item = ChannelUpdate<M::ChannelKey, M::ChannelPayload>>,
    ) -> Result<()>;

    fn active_channel_count(&self) -> usize;
}
