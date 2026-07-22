use std::marker::PhantomData;

use rand::Rng;

use crate::{
    Error, Result,
    engine::FamilySimulation,
    scheduler::{
        FamilyAlgorithm, FamilyUpdateContext, ScheduledFamilyChannel,
        family::{
            FamilyBundle, FamilyCons, FamilyId, FamilyNil, FamilyReactionSet, StaticFamilyModel,
            StaticReactionFamily,
        },
        math::checked_propensity,
        nrm_clock::FamilyClocks,
        update::{
            FamilyChannelUpdate, FamilyChannelUpdates, FamilyUpdateBuffer, FamilyUpdateKind,
            InlineFamilyChannelUpdates,
        },
    },
};

/// Family-separated NRM algorithm backed by a vector of indexed clock heaps.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct FamilyNrmAlgorithm {
    clocks: Vec<FamilyClocks>,
}

/// Readable family-separated Next Reaction Method simulation.
///
/// Every concrete channel stores its current propensity and absolute firing time. Only the fired
/// channel, explicitly reported dependencies, and newly appended channels receive scheduler work.
pub type FamilyNrm<M, F> = FamilySimulation<M, F, FamilyNrmAlgorithm, FamilyChannelUpdates>;

impl<M, F> FamilySimulation<M, F, FamilyNrmAlgorithm, FamilyChannelUpdates>
where
    M: StaticFamilyModel<Families = F>,
    F: FamilyReactionSet<M>,
{
    /// Initialize one NRM clock for every concrete channel at time zero.
    pub fn new<R: Rng + ?Sized>(model: M, families: F, clock_rng: &mut R) -> Result<Self> {
        let state = model.initial_state();
        let algorithm = FamilyNrmAlgorithm::new(&model, &state, &families, clock_rng)?;
        Ok(Self::from_initialized(
            model,
            state,
            families,
            algorithm,
            FamilyChannelUpdates::default(),
        ))
    }
}

impl FamilyNrmAlgorithm {
    fn new<M, F, R>(model: &M, state: &M::State, families: &F, rng: &mut R) -> Result<Self>
    where
        M: StaticFamilyModel<Families = F>,
        F: FamilyReactionSet<M>,
        R: Rng + ?Sized,
    {
        let mut clocks = Vec::with_capacity(F::FAMILY_COUNT);
        for family in 0..F::FAMILY_COUNT {
            let mut family_clocks = FamilyClocks::default();
            let channel_count =
                families
                    .channel_count(model, state, family)
                    .ok_or(Error::InvalidFamily {
                        family,
                        family_count: F::FAMILY_COUNT,
                    })?;
            family_clocks.resize_with(channel_count, 0.0, rng, |local_channel| {
                let propensity = families
                    .propensity(model, state, family, local_channel, 0.0)
                    .ok_or(Error::InvalidFamily {
                        family,
                        family_count: F::FAMILY_COUNT,
                    })?;
                checked_propensity(propensity)
            })?;
            clocks.push(family_clocks);
        }
        Ok(Self { clocks })
    }

    fn resize_clocks<M, F, R>(
        &mut self,
        model: &M,
        state: &M::State,
        families: &F,
        time: f64,
        rng: &mut R,
    ) -> Result<()>
    where
        M: StaticFamilyModel<Families = F>,
        F: FamilyReactionSet<M>,
        R: Rng + ?Sized,
    {
        for family in 0..F::FAMILY_COUNT {
            let channel_count =
                families
                    .channel_count(model, state, family)
                    .ok_or(Error::InvalidFamily {
                        family,
                        family_count: F::FAMILY_COUNT,
                    })?;
            self.clocks[family].resize_with(channel_count, time, rng, |local_channel| {
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

    fn next_scheduled_channel<F>(&self) -> Option<ScheduledFamilyChannel<F>> {
        self.clocks
            .iter()
            .enumerate()
            .filter_map(|(family, clocks)| {
                let (local_channel, time) = clocks.peek()?;
                Some(ScheduledFamilyChannel {
                    // SAFETY: `family` comes from enumerating the configured family clocks.
                    family: unsafe { FamilyId::__from_index_unchecked(family) },
                    local_channel,
                    time,
                })
            })
            .min_by(|left, right| left.time.total_cmp(&right.time))
    }

    fn validate_channel<M, F>(&self, family: usize, local_channel: usize) -> Result<()>
    where
        M: StaticFamilyModel<Families = F>,
        F: FamilyReactionSet<M>,
    {
        let Some(clocks) = self.clocks.get(family) else {
            return Err(Error::InvalidFamily {
                family,
                family_count: F::FAMILY_COUNT,
            });
        };
        if local_channel >= clocks.len() {
            return Err(Error::InvalidFamilyChannel {
                family,
                local_channel,
                channel_count: clocks.len(),
            });
        }
        Ok(())
    }

    fn reschedule<M, F, R>(
        &mut self,
        context: &mut FamilyUpdateContext<'_, M, F, R>,
        update: FamilyChannelUpdate,
    ) -> Result<()>
    where
        M: StaticFamilyModel<Families = F>,
        F: FamilyReactionSet<M>,
        R: Rng + ?Sized,
    {
        self.validate_channel::<M, F>(update.family, update.local_channel)?;
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
        self.clocks[update.family].reschedule(
            update.local_channel,
            checked_propensity(propensity)?,
            context.time,
            update.kind == FamilyUpdateKind::Reinitialize,
            context.clock_rng,
        );
        Ok(())
    }
}

impl<M, F> FamilyAlgorithm<M, F> for FamilyNrmAlgorithm
where
    M: StaticFamilyModel<Families = F>,
    F: FamilyReactionSet<M>,
{
    fn next_event<C, S>(
        &mut self,
        _now: f64,
        _clock_rng: &mut C,
        _selection_rng: &mut S,
    ) -> Result<Option<ScheduledFamilyChannel<F>>>
    where
        C: Rng + ?Sized,
        S: Rng + ?Sized,
    {
        Ok(self.next_scheduled_channel())
    }

    fn apply_updates<C, U>(
        &mut self,
        mut context: FamilyUpdateContext<'_, M, F, C>,
        updates: &mut U,
    ) -> Result<()>
    where
        C: Rng + ?Sized,
        U: FamilyUpdateBuffer,
    {
        self.resize_clocks(
            context.model,
            context.state,
            context.families,
            context.time,
            context.clock_rng,
        )?;

        let fired = context.fired;
        let fired_family = fired.family.index();
        let fired_still_exists = self
            .clocks
            .get(fired_family)
            .is_some_and(|clocks| fired.local_channel < clocks.len());
        if fired_still_exists {
            self.reschedule(&mut context, FamilyChannelUpdate {
                family: fired_family,
                local_channel: fired.local_channel,
                kind: FamilyUpdateKind::Reinitialize,
            })?;
        }

        while let Some(update) = updates.pop() {
            self.validate_channel::<M, F>(update.family, update.local_channel)?;
            if update.family == fired_family && update.local_channel == fired.local_channel {
                continue;
            }
            self.reschedule(&mut context, update)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default)]
#[doc(hidden)]
pub struct StaticClockNil;

#[derive(Debug, Clone)]
#[doc(hidden)]
pub struct StaticClockCons<T> {
    head: FamilyClocks,
    tail: T,
}

#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StaticRescheduleOutcome {
    Updated,
    MissingFamily,
    MissingChannel { channel_count: usize },
}

mod sealed {
    pub trait StaticNrmFamilySet {}
}

impl sealed::StaticNrmFamilySet for FamilyNil {}

impl<H, T> sealed::StaticNrmFamilySet for FamilyCons<H, T> where T: sealed::StaticNrmFamilySet {}

impl<Tag, L> sealed::StaticNrmFamilySet for FamilyBundle<Tag, L> where L: sealed::StaticNrmFamilySet {}

/// Static NRM clock-storage operations for an arbitrary-length reaction-family list.
#[doc(hidden)]
pub trait StaticNrmFamilySet<M>: FamilyReactionSet<M> + sealed::StaticNrmFamilySet
where
    M: StaticFamilyModel,
{
    type Clocks;

    fn initialize_clocks<R: Rng + ?Sized>(
        &self,
        model: &M,
        state: &M::State,
        time: f64,
        rng: &mut R,
    ) -> Result<Self::Clocks>;

    fn resize_clocks<R: Rng + ?Sized>(
        &self,
        clocks: &mut Self::Clocks,
        model: &M,
        state: &M::State,
        time: f64,
        rng: &mut R,
    ) -> Result<()>;

    fn next_scheduled_channel(
        clocks: &Self::Clocks,
        family_offset: usize,
    ) -> Option<(usize, usize, f64)>;

    fn reschedule_channel<R: Rng + ?Sized>(
        &self,
        clocks: &mut Self::Clocks,
        model: &M,
        state: &M::State,
        time: f64,
        update: FamilyChannelUpdate,
        rng: &mut R,
    ) -> Result<StaticRescheduleOutcome>;
}

impl<M> StaticNrmFamilySet<M> for FamilyNil
where
    M: StaticFamilyModel,
{
    type Clocks = StaticClockNil;

    fn initialize_clocks<R: Rng + ?Sized>(
        &self,
        _model: &M,
        _state: &M::State,
        _time: f64,
        _rng: &mut R,
    ) -> Result<Self::Clocks> {
        Ok(StaticClockNil)
    }

    fn resize_clocks<R: Rng + ?Sized>(
        &self,
        _clocks: &mut Self::Clocks,
        _model: &M,
        _state: &M::State,
        _time: f64,
        _rng: &mut R,
    ) -> Result<()> {
        Ok(())
    }

    fn next_scheduled_channel(
        _clocks: &Self::Clocks,
        _family_offset: usize,
    ) -> Option<(usize, usize, f64)> {
        None
    }

    fn reschedule_channel<R: Rng + ?Sized>(
        &self,
        _clocks: &mut Self::Clocks,
        _model: &M,
        _state: &M::State,
        _time: f64,
        _update: FamilyChannelUpdate,
        _rng: &mut R,
    ) -> Result<StaticRescheduleOutcome> {
        Ok(StaticRescheduleOutcome::MissingFamily)
    }
}

impl<M, H, T> StaticNrmFamilySet<M> for FamilyCons<H, T>
where
    M: StaticFamilyModel,
    H: StaticReactionFamily<M>,
    T: StaticNrmFamilySet<M>,
{
    type Clocks = StaticClockCons<T::Clocks>;

    fn initialize_clocks<R: Rng + ?Sized>(
        &self,
        model: &M,
        state: &M::State,
        time: f64,
        rng: &mut R,
    ) -> Result<Self::Clocks> {
        let mut head = FamilyClocks::default();
        head.resize_with(
            self.head.channel_count(model, state),
            time,
            rng,
            |local_channel| {
                checked_propensity(self.head.propensity(model, state, local_channel, time))
            },
        )?;
        let tail = self.tail.initialize_clocks(model, state, time, rng)?;
        Ok(StaticClockCons { head, tail })
    }

    fn resize_clocks<R: Rng + ?Sized>(
        &self,
        clocks: &mut Self::Clocks,
        model: &M,
        state: &M::State,
        time: f64,
        rng: &mut R,
    ) -> Result<()> {
        clocks.head.resize_with(
            self.head.channel_count(model, state),
            time,
            rng,
            |local_channel| {
                checked_propensity(self.head.propensity(model, state, local_channel, time))
            },
        )?;
        self.tail
            .resize_clocks(&mut clocks.tail, model, state, time, rng)
    }

    fn next_scheduled_channel(
        clocks: &Self::Clocks,
        family_offset: usize,
    ) -> Option<(usize, usize, f64)> {
        let head = clocks
            .head
            .peek()
            .map(|(local_channel, time)| (family_offset, local_channel, time));
        let tail = T::next_scheduled_channel(&clocks.tail, family_offset + 1);
        match (head, tail) {
            (Some(left), Some(right)) => {
                if left.2.total_cmp(&right.2).is_le() {
                    Some(left)
                } else {
                    Some(right)
                }
            }
            (Some(channel), None) | (None, Some(channel)) => Some(channel),
            (None, None) => None,
        }
    }

    fn reschedule_channel<R: Rng + ?Sized>(
        &self,
        clocks: &mut Self::Clocks,
        model: &M,
        state: &M::State,
        time: f64,
        update: FamilyChannelUpdate,
        rng: &mut R,
    ) -> Result<StaticRescheduleOutcome> {
        if update.family == 0 {
            if update.local_channel >= clocks.head.len() {
                return Ok(StaticRescheduleOutcome::MissingChannel {
                    channel_count: clocks.head.len(),
                });
            }
            let propensity =
                checked_propensity(
                    self.head
                        .propensity(model, state, update.local_channel, time),
                )?;
            clocks.head.reschedule(
                update.local_channel,
                propensity,
                time,
                update.kind == FamilyUpdateKind::Reinitialize,
                rng,
            );
            Ok(StaticRescheduleOutcome::Updated)
        } else {
            self.tail.reschedule_channel(
                &mut clocks.tail,
                model,
                state,
                time,
                FamilyChannelUpdate {
                    family: update.family - 1,
                    ..update
                },
                rng,
            )
        }
    }
}

impl<M, Tag, L> StaticNrmFamilySet<M> for FamilyBundle<Tag, L>
where
    M: StaticFamilyModel,
    L: StaticNrmFamilySet<M>,
{
    type Clocks = L::Clocks;

    #[inline(always)]
    fn initialize_clocks<R: Rng + ?Sized>(
        &self,
        model: &M,
        state: &M::State,
        time: f64,
        rng: &mut R,
    ) -> Result<Self::Clocks> {
        self.families.initialize_clocks(model, state, time, rng)
    }

    #[inline(always)]
    fn resize_clocks<R: Rng + ?Sized>(
        &self,
        clocks: &mut Self::Clocks,
        model: &M,
        state: &M::State,
        time: f64,
        rng: &mut R,
    ) -> Result<()> {
        self.families.resize_clocks(clocks, model, state, time, rng)
    }

    #[inline(always)]
    fn next_scheduled_channel(
        clocks: &Self::Clocks,
        family_offset: usize,
    ) -> Option<(usize, usize, f64)> {
        L::next_scheduled_channel(clocks, family_offset)
    }

    #[inline(always)]
    fn reschedule_channel<R: Rng + ?Sized>(
        &self,
        clocks: &mut Self::Clocks,
        model: &M,
        state: &M::State,
        time: f64,
        update: FamilyChannelUpdate,
        rng: &mut R,
    ) -> Result<StaticRescheduleOutcome> {
        self.families
            .reschedule_channel(clocks, model, state, time, update, rng)
    }
}

/// Statically stored NRM algorithm for an arbitrary-length family list.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct StaticFamilyNrmAlgorithm<M, F>
where
    M: StaticFamilyModel<Families = F>,
    F: StaticNrmFamilySet<M>,
{
    clocks: F::Clocks,
    model: PhantomData<fn() -> M>,
}

/// Statically dispatched family-separated Next Reaction Method simulation.
///
/// `INLINE_CAP` controls how many dependency updates stay inline. Larger event fanout spills into a
/// reusable heap buffer instead of panicking.
pub type StaticFamilyNrm<M, F, const INLINE_CAP: usize> =
    FamilySimulation<M, F, StaticFamilyNrmAlgorithm<M, F>, InlineFamilyChannelUpdates<INLINE_CAP>>;

impl<M, F, const INLINE_CAP: usize>
    FamilySimulation<M, F, StaticFamilyNrmAlgorithm<M, F>, InlineFamilyChannelUpdates<INLINE_CAP>>
where
    M: StaticFamilyModel<Families = F>,
    F: StaticNrmFamilySet<M>,
{
    /// Initialize statically stored NRM clocks for every concrete channel at time zero.
    pub fn new<R: Rng + ?Sized>(model: M, families: F, clock_rng: &mut R) -> Result<Self> {
        let state = model.initial_state();
        let clocks = families.initialize_clocks(&model, &state, 0.0, clock_rng)?;
        let algorithm = StaticFamilyNrmAlgorithm {
            clocks,
            model: PhantomData,
        };
        Ok(Self::from_initialized(
            model,
            state,
            families,
            algorithm,
            InlineFamilyChannelUpdates::default(),
        ))
    }
}

impl<M, F> StaticFamilyNrmAlgorithm<M, F>
where
    M: StaticFamilyModel<Families = F>,
    F: StaticNrmFamilySet<M>,
{
    fn reschedule<R: Rng + ?Sized>(
        &mut self,
        context: &mut FamilyUpdateContext<'_, M, F, R>,
        update: FamilyChannelUpdate,
    ) -> Result<StaticRescheduleOutcome> {
        context.families.reschedule_channel(
            &mut self.clocks,
            context.model,
            context.state,
            context.time,
            update,
            context.clock_rng,
        )
    }

    fn require_reschedule<R: Rng + ?Sized>(
        &mut self,
        context: &mut FamilyUpdateContext<'_, M, F, R>,
        update: FamilyChannelUpdate,
    ) -> Result<()> {
        match self.reschedule(context, update)? {
            StaticRescheduleOutcome::Updated => Ok(()),
            StaticRescheduleOutcome::MissingFamily => Err(Error::InvalidFamily {
                family: update.family,
                family_count: F::FAMILY_COUNT,
            }),
            StaticRescheduleOutcome::MissingChannel { channel_count } => {
                Err(Error::InvalidFamilyChannel {
                    family: update.family,
                    local_channel: update.local_channel,
                    channel_count,
                })
            }
        }
    }
}

impl<M, F> FamilyAlgorithm<M, F> for StaticFamilyNrmAlgorithm<M, F>
where
    M: StaticFamilyModel<Families = F>,
    F: StaticNrmFamilySet<M>,
{
    fn next_event<C, S>(
        &mut self,
        _now: f64,
        _clock_rng: &mut C,
        _selection_rng: &mut S,
    ) -> Result<Option<ScheduledFamilyChannel<F>>>
    where
        C: Rng + ?Sized,
        S: Rng + ?Sized,
    {
        Ok(
            F::next_scheduled_channel(&self.clocks, 0).map(|(family, local_channel, time)| {
                ScheduledFamilyChannel {
                    // SAFETY: static clock traversal follows the configured family-list ordering.
                    family: unsafe { FamilyId::__from_index_unchecked(family) },
                    local_channel,
                    time,
                }
            }),
        )
    }

    fn apply_updates<C, U>(
        &mut self,
        mut context: FamilyUpdateContext<'_, M, F, C>,
        updates: &mut U,
    ) -> Result<()>
    where
        C: Rng + ?Sized,
        U: FamilyUpdateBuffer,
    {
        context.families.resize_clocks(
            &mut self.clocks,
            context.model,
            context.state,
            context.time,
            context.clock_rng,
        )?;

        let fired = context.fired;
        let fired_family = fired.family.index();
        let fired_still_exists = match self.reschedule(&mut context, FamilyChannelUpdate {
            family: fired_family,
            local_channel: fired.local_channel,
            kind: FamilyUpdateKind::Reinitialize,
        })? {
            StaticRescheduleOutcome::Updated => true,
            StaticRescheduleOutcome::MissingChannel { .. } => false,
            StaticRescheduleOutcome::MissingFamily => {
                return Err(Error::InvalidFamily {
                    family: fired_family,
                    family_count: F::FAMILY_COUNT,
                });
            }
        };

        while let Some(update) = updates.pop() {
            if fired_still_exists
                && update.family == fired_family
                && update.local_channel == fired.local_channel
            {
                continue;
            }
            self.require_reschedule(&mut context, update)?;
        }
        Ok(())
    }
}
