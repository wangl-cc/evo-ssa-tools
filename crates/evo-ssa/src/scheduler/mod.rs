/// Cached Direct Method schedulers for dynamic channels and static families.
pub mod direct;
/// Arbitrary-length static reaction-family model contracts and construction macros.
pub mod family;
mod math;
/// Next Reaction Method schedulers for static reaction families.
pub mod nrm;
mod nrm_clock;
/// Typed dependency-update sinks for static reaction families.
pub mod update;
mod weighted;

use rand::Rng;

use crate::{
    Result,
    model::{ChannelUpdate, EvolvingModel},
    scheduler::{
        family::{FamilyId, FamilyReactionSet, StaticFamilyModel},
        update::FamilyUpdateBuffer,
    },
};

/// A concrete channel selected by a scheduler.
#[derive(Debug, Clone)]
pub struct ScheduledEvent<K, P> {
    /// Model-defined identity of the selected channel.
    pub key: K,
    /// Cached payload associated with the selected channel.
    pub payload: P,
    /// Waiting time since the previous event.
    pub dt: f64,
}

/// Common interface for exact SSA schedulers.
pub trait SsaScheduler<M: EvolvingModel> {
    /// Replace scheduler state with the model's initial channel set.
    fn initialize(
        &mut self,
        model: &M,
        state: &M::State,
        time: f64,
        channels: impl IntoIterator<Item = ChannelUpdate<M::ChannelKey, M::ChannelPayload>>,
    ) -> Result<()>;

    /// Sample the next channel and waiting time, or return `None` when all propensities are zero.
    fn next_event<C, S>(
        &mut self,
        clock_rng: &mut C,
        selection_rng: &mut S,
    ) -> Result<Option<ScheduledEvent<M::ChannelKey, M::ChannelPayload>>>
    where
        C: Rng + ?Sized,
        S: Rng + ?Sized;

    /// Apply model-reported channel changes after an event.
    fn apply_updates(
        &mut self,
        model: &M,
        state: &M::State,
        time: f64,
        updates: impl IntoIterator<Item = ChannelUpdate<M::ChannelKey, M::ChannelPayload>>,
    ) -> Result<()>;

    /// Return the number of currently active concrete channels.
    fn active_channel_count(&self) -> usize;
}

#[doc(hidden)]
pub struct ScheduledFamilyChannel<F> {
    pub(crate) family: FamilyId<F>,
    pub(crate) local_channel: usize,
    pub(crate) time: f64,
}

impl<F> Clone for ScheduledFamilyChannel<F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F> Copy for ScheduledFamilyChannel<F> {}

#[doc(hidden)]
pub struct FamilyUpdateContext<'a, M, F, C>
where
    M: StaticFamilyModel<Families = F>,
    C: ?Sized,
{
    pub(crate) model: &'a M,
    pub(crate) state: &'a M::State,
    pub(crate) families: &'a F,
    pub(crate) time: f64,
    pub(crate) fired: ScheduledFamilyChannel<F>,
    pub(crate) clock_rng: &'a mut C,
}

#[doc(hidden)]
pub trait FamilyAlgorithm<M, F>
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
        S: Rng + ?Sized;

    fn apply_updates<C, U>(
        &mut self,
        context: FamilyUpdateContext<'_, M, F, C>,
        updates: &mut U,
    ) -> Result<()>
    where
        C: Rng + ?Sized,
        U: FamilyUpdateBuffer;
}
