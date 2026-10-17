use rand::Rng;

use crate::{
    Error, Result,
    model::{ChannelEditor, EvolvingModel},
    random::SsaRngs,
    scheduler::{
        FamilyAlgorithm, FamilyUpdateContext, SsaScheduler,
        direct::DirectScheduler,
        family::{FamilyReactionSet, StaticFamilyModel},
        update::{FamilyLocalUpdateWriter, FamilyUpdateBuffer},
    },
};

/// A running simulation with owned model state and scheduler state.
#[derive(Debug, Clone)]
pub struct Simulation<M, S>
where
    M: EvolvingModel,
{
    model: M,
    state: M::State,
    scheduler: S,
    time: f64,
    poisoned: bool,
}

impl<M> Simulation<M, DirectScheduler<M::ChannelKey, M::ChannelPayload>>
where
    M: EvolvingModel,
{
    /// Create a simulation using the cached Direct Method scheduler.
    pub fn direct(model: M) -> Result<Self> {
        Self::with_scheduler(model, DirectScheduler::new())
    }
}

impl<M, S> Simulation<M, S>
where
    M: EvolvingModel,
    S: SsaScheduler<M>,
{
    /// Create a simulation from a model and compatible scheduler.
    pub fn with_scheduler(model: M, mut scheduler: S) -> Result<Self> {
        let state = model.initial_state();
        let time = 0.0;
        let mut channels = ChannelEditor::new();
        model.initialize_channels(&state, &mut channels);
        scheduler.initialize(&model, &state, time, channels.into_updates())?;

        Ok(Self {
            model,
            state,
            scheduler,
            time,
            poisoned: false,
        })
    }

    /// Return the current simulation time.
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Borrow the current model state.
    pub fn state(&self) -> &M::State {
        &self.state
    }

    /// Borrow the model definition.
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Borrow the installed scheduler and its cached state.
    pub fn scheduler(&self) -> &S {
        &self.scheduler
    }

    /// Execute one SSA event.
    pub fn step<C, Q, E>(&mut self, rngs: &mut SsaRngs<C, Q, E>) -> Result<Option<M::Event>>
    where
        C: Rng,
        Q: Rng,
        E: Rng,
    {
        if self.poisoned {
            return Err(Error::SimulationPoisoned);
        }

        let (clock_rng, selection_rng, event_rng) = rngs.streams();
        let Some(next) = self.scheduler.next_event(clock_rng, selection_rng)? else {
            return Ok(None);
        };

        self.time += next.dt;
        let event = self
            .model
            .fire(&mut self.state, next.key, &next.payload, event_rng);

        let mut updates = ChannelEditor::new();
        self.model
            .refresh_after_event(&self.state, &event, &mut updates);
        if let Err(error) = self.scheduler.apply_updates(
            &self.model,
            &self.state,
            self.time,
            updates.into_updates(),
        ) {
            self.poisoned = true;
            return Err(error);
        }

        Ok(Some(event))
    }

    /// Run until `max_events` has fired or no active channel has positive propensity.
    pub fn run<C, Q, E>(
        &mut self,
        rngs: &mut SsaRngs<C, Q, E>,
        max_events: usize,
    ) -> Result<RunStatus>
    where
        C: Rng,
        Q: Rng,
        E: Rng,
    {
        let mut fired_events = 0;
        while fired_events < max_events {
            if self.step(rngs)?.is_none() {
                return Ok(RunStatus::NoActiveChannels { fired_events });
            }
            fired_events += 1;
        }

        Ok(RunStatus::MaxEvents { fired_events })
    }
}

/// A running static-family simulation with a monomorphized scheduling algorithm.
///
/// Public scheduler aliases such as [`FamilyDirect`](crate::scheduler::direct::FamilyDirect) and
/// [`FamilyNrm`](crate::scheduler::nrm::FamilyNrm) select the algorithm and update-buffer strategy.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct FamilySimulation<M, F, A, U>
where
    M: StaticFamilyModel,
{
    model: M,
    state: M::State,
    families: F,
    algorithm: A,
    updates: U,
    time: f64,
    poisoned: bool,
}

impl<M, F, A, U> FamilySimulation<M, F, A, U>
where
    M: StaticFamilyModel<Families = F>,
    F: FamilyReactionSet<M>,
    A: FamilyAlgorithm<M, F>,
    U: FamilyUpdateBuffer,
{
    pub(crate) fn from_initialized(
        model: M,
        state: M::State,
        families: F,
        algorithm: A,
        updates: U,
    ) -> Self {
        Self {
            model,
            state,
            families,
            algorithm,
            updates,
            time: 0.0,
            poisoned: false,
        }
    }

    /// Return the current simulation time.
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Borrow the current model state.
    pub fn state(&self) -> &M::State {
        &self.state
    }

    /// Borrow the model definition.
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Borrow the compile-time family bundle.
    pub fn families(&self) -> &F {
        &self.families
    }

    /// Execute one event and return its model-defined description.
    pub fn step<C, S, E>(&mut self, rngs: &mut SsaRngs<C, S, E>) -> Result<Option<M::Event>>
    where
        C: Rng,
        S: Rng,
        E: Rng,
    {
        if self.poisoned {
            return Err(Error::SimulationPoisoned);
        }

        let (clock_rng, selection_rng, event_rng) = rngs.streams();
        let Some(fired) = self
            .algorithm
            .next_event(self.time, clock_rng, selection_rng)?
        else {
            return Ok(None);
        };

        self.time = fired.time;
        let event = self
            .families
            .fire(
                &self.model,
                &mut self.state,
                fired.family.index(),
                fired.local_channel,
                event_rng,
            )
            .ok_or(Error::MissingSelectedChannel)?;

        self.updates.clear();
        {
            let current_family = fired.family;
            let mut writer = FamilyLocalUpdateWriter::new(current_family, &mut self.updates);
            self.model
                .refresh_after_event(&self.state, &event, &mut writer);
        }

        let context = FamilyUpdateContext {
            model: &self.model,
            state: &self.state,
            families: &self.families,
            time: self.time,
            fired,
            clock_rng,
        };
        if let Err(error) = self.algorithm.apply_updates(context, &mut self.updates) {
            self.poisoned = true;
            return Err(error);
        }

        Ok(Some(event))
    }

    /// Run until `max_events` has fired or no active channel has positive propensity.
    pub fn run<C, S, E>(
        &mut self,
        rngs: &mut SsaRngs<C, S, E>,
        max_events: usize,
    ) -> Result<RunStatus>
    where
        C: Rng,
        S: Rng,
        E: Rng,
    {
        let mut fired_events = 0;
        while fired_events < max_events {
            if self.step(rngs)?.is_none() {
                return Ok(RunStatus::NoActiveChannels { fired_events });
            }
            fired_events += 1;
        }
        Ok(RunStatus::MaxEvents { fired_events })
    }
}

/// Reason a bounded simulation run returned to its caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunStatus {
    /// The requested event limit was reached.
    MaxEvents {
        /// Number of events fired during this call to `run`.
        fired_events: usize,
    },
    /// No channel with positive propensity remained.
    NoActiveChannels {
        /// Number of events fired during this call to `run`.
        fired_events: usize,
    },
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use rand::{SeedableRng, rngs::SmallRng};

    use super::*;
    use crate::scheduler::direct::DirectSchedulerStats;

    fn rngs(seed: u64) -> SsaRngs<SmallRng, SmallRng, SmallRng> {
        SsaRngs::new(
            SmallRng::seed_from_u64(seed),
            SmallRng::seed_from_u64(seed + 1),
            SmallRng::seed_from_u64(seed + 2),
        )
    }

    #[derive(Debug, Clone)]
    struct MultiSpeciesBirthDeath {
        species_count: usize,
        birth_rate: f64,
        death_rate: f64,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    enum BirthDeathChannel {
        Birth(usize),
        Death(usize),
    }

    #[derive(Debug, Clone, Copy)]
    struct SpeciesPayload {
        species: usize,
    }

    #[derive(Debug, Clone, Copy)]
    struct BirthDeathEvent {
        species: usize,
    }

    impl EvolvingModel for MultiSpeciesBirthDeath {
        type ChannelKey = BirthDeathChannel;
        type ChannelPayload = SpeciesPayload;
        type Event = BirthDeathEvent;
        type State = Vec<u64>;

        fn initial_state(&self) -> Self::State {
            vec![1; self.species_count]
        }

        fn initialize_channels(
            &self,
            state: &Self::State,
            channels: &mut ChannelEditor<Self::ChannelKey, Self::ChannelPayload>,
        ) {
            for species in 0..state.len() {
                let payload = SpeciesPayload { species };
                channels.upsert(BirthDeathChannel::Birth(species), payload);
                channels.upsert(BirthDeathChannel::Death(species), payload);
            }
        }

        fn propensity(
            &self,
            state: &Self::State,
            key: Self::ChannelKey,
            payload: &Self::ChannelPayload,
            _time: f64,
        ) -> f64 {
            let count = state[payload.species] as f64;
            match key {
                BirthDeathChannel::Birth(_) => self.birth_rate * count,
                BirthDeathChannel::Death(_) => self.death_rate * count,
            }
        }

        fn fire<R: Rng + ?Sized>(
            &self,
            state: &mut Self::State,
            key: Self::ChannelKey,
            payload: &Self::ChannelPayload,
            _rng: &mut R,
        ) -> Self::Event {
            match key {
                BirthDeathChannel::Birth(_) => state[payload.species] += 1,
                BirthDeathChannel::Death(_) => {
                    state[payload.species] = state[payload.species].saturating_sub(1);
                }
            }
            BirthDeathEvent {
                species: payload.species,
            }
        }

        fn refresh_after_event(
            &self,
            _state: &Self::State,
            event: &Self::Event,
            channels: &mut ChannelEditor<Self::ChannelKey, Self::ChannelPayload>,
        ) {
            channels.recompute(BirthDeathChannel::Birth(event.species));
            channels.recompute(BirthDeathChannel::Death(event.species));
        }
    }

    #[test]
    fn direct_simulation_accepts_many_concrete_channels() -> Result<()> {
        let model = MultiSpeciesBirthDeath {
            species_count: 128,
            birth_rate: 1.0,
            death_rate: 0.25,
        };
        let mut simulation = Simulation::direct(model)?;
        let stats: DirectSchedulerStats = simulation.scheduler().stats();

        assert_eq!(stats.active_channel_count, 256);
        assert_eq!(stats.allocated_slot_count, 256);
        assert_eq!(stats.total_propensity, 160.0);

        let mut rngs = rngs(7);
        let status = simulation.run(&mut rngs, 32)?;

        assert_eq!(status, RunStatus::MaxEvents { fired_events: 32 });
        assert!(simulation.time() > 0.0);
        Ok(())
    }

    #[derive(Debug, Clone)]
    struct ChainMutation {
        max_species: usize,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    enum MutationChannel {
        Mutate(usize),
    }

    #[derive(Debug, Clone, Copy)]
    struct MutationPayload {
        source: usize,
    }

    #[derive(Debug, Clone, Copy)]
    struct MutationEvent {
        source: usize,
        child: Option<usize>,
    }

    impl EvolvingModel for ChainMutation {
        type ChannelKey = MutationChannel;
        type ChannelPayload = MutationPayload;
        type Event = MutationEvent;
        type State = Vec<u64>;

        fn initial_state(&self) -> Self::State {
            vec![1]
        }

        fn initialize_channels(
            &self,
            state: &Self::State,
            channels: &mut ChannelEditor<Self::ChannelKey, Self::ChannelPayload>,
        ) {
            for source in 0..state.len() {
                channels.upsert(MutationChannel::Mutate(source), MutationPayload { source });
            }
        }

        fn propensity(
            &self,
            state: &Self::State,
            _key: Self::ChannelKey,
            payload: &Self::ChannelPayload,
            _time: f64,
        ) -> f64 {
            if payload.source + 1 < self.max_species {
                state[payload.source] as f64
            } else {
                0.0
            }
        }

        fn fire<R: Rng + ?Sized>(
            &self,
            state: &mut Self::State,
            _key: Self::ChannelKey,
            payload: &Self::ChannelPayload,
            _rng: &mut R,
        ) -> Self::Event {
            state[payload.source] -= 1;
            let child = payload.source + 1;
            if child == state.len() {
                state.push(0);
            }
            state[child] += 1;

            MutationEvent {
                source: payload.source,
                child: Some(child),
            }
        }

        fn refresh_after_event(
            &self,
            _state: &Self::State,
            event: &Self::Event,
            channels: &mut ChannelEditor<Self::ChannelKey, Self::ChannelPayload>,
        ) {
            channels.recompute(MutationChannel::Mutate(event.source));
            if let Some(child) = event.child {
                channels.upsert(MutationChannel::Mutate(child), MutationPayload {
                    source: child,
                });
            }
        }
    }

    #[test]
    fn direct_simulation_can_add_channels_after_species_creation() -> Result<()> {
        let model = ChainMutation { max_species: 4 };
        let mut simulation = Simulation::direct(model)?;
        assert_eq!(simulation.scheduler().stats().active_channel_count, 1);

        let mut rngs = rngs(11);
        simulation.step(&mut rngs)?;

        assert_eq!(simulation.state(), &[0, 1]);
        assert_eq!(simulation.scheduler().stats().active_channel_count, 2);
        assert_eq!(simulation.scheduler().stats().total_propensity, 1.0);
        Ok(())
    }
}
