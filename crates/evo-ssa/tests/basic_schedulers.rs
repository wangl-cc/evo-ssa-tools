use evo_ssa::prelude::*;
use rand::{Rng, SeedableRng, rngs::SmallRng};

#[derive(Debug, Clone, Copy)]
struct DynamicBirthDeath {
    initial: u64,
    birth_rate: f64,
    death_rate: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum DynamicChannel {
    Birth,
    Death,
}

#[derive(Debug, Clone, Copy)]
struct DynamicEvent;

impl EvolvingModel for DynamicBirthDeath {
    type ChannelKey = DynamicChannel;
    type ChannelPayload = DynamicChannel;
    type Event = DynamicEvent;
    type State = u64;

    fn initial_state(&self) -> Self::State {
        self.initial
    }

    fn initialize_channels(
        &self,
        _state: &Self::State,
        channels: &mut ChannelEditor<Self::ChannelKey, Self::ChannelPayload>,
    ) {
        channels.upsert(DynamicChannel::Birth, DynamicChannel::Birth);
        channels.upsert(DynamicChannel::Death, DynamicChannel::Death);
    }

    fn propensity(
        &self,
        state: &Self::State,
        key: Self::ChannelKey,
        _payload: &Self::ChannelPayload,
        _time: f64,
    ) -> f64 {
        match key {
            DynamicChannel::Birth => self.birth_rate * (*state as f64 + 1.0),
            DynamicChannel::Death => self.death_rate * *state as f64,
        }
    }

    fn fire<R: Rng + ?Sized>(
        &self,
        state: &mut Self::State,
        key: Self::ChannelKey,
        _payload: &Self::ChannelPayload,
        _rng: &mut R,
    ) -> Self::Event {
        match key {
            DynamicChannel::Birth => *state += 1,
            DynamicChannel::Death => *state = state.saturating_sub(1),
        }
        DynamicEvent
    }

    fn refresh_after_event(
        &self,
        _state: &Self::State,
        _event: &Self::Event,
        channels: &mut ChannelEditor<Self::ChannelKey, Self::ChannelPayload>,
    ) {
        channels.recompute(DynamicChannel::Birth);
        channels.recompute(DynamicChannel::Death);
    }
}

#[derive(Debug, Clone, Copy)]
struct DenseBirthDeath(DynamicBirthDeath);

impl EvolvingModel for DenseBirthDeath {
    type ChannelKey = usize;
    type ChannelPayload = usize;
    type Event = DynamicEvent;
    type State = u64;

    fn initial_state(&self) -> Self::State {
        self.0.initial
    }

    fn initialize_channels(
        &self,
        _state: &Self::State,
        channels: &mut ChannelEditor<Self::ChannelKey, Self::ChannelPayload>,
    ) {
        channels.upsert(0, 0);
        channels.upsert(1, 1);
    }

    fn propensity(
        &self,
        state: &Self::State,
        key: Self::ChannelKey,
        _payload: &Self::ChannelPayload,
        _time: f64,
    ) -> f64 {
        match key {
            0 => self.0.birth_rate * (*state as f64 + 1.0),
            1 => self.0.death_rate * *state as f64,
            _ => 0.0,
        }
    }

    fn fire<R: Rng + ?Sized>(
        &self,
        state: &mut Self::State,
        key: Self::ChannelKey,
        _payload: &Self::ChannelPayload,
        _rng: &mut R,
    ) -> Self::Event {
        match key {
            0 => *state += 1,
            1 => *state = state.saturating_sub(1),
            _ => {}
        }
        DynamicEvent
    }

    fn refresh_after_event(
        &self,
        _state: &Self::State,
        _event: &Self::Event,
        channels: &mut ChannelEditor<Self::ChannelKey, Self::ChannelPayload>,
    ) {
        channels.recompute(0);
        channels.recompute(1);
    }
}

#[test]
fn dynamic_direct_schedulers_run_basic_birth_death_model() -> Result<()> {
    let model = DynamicBirthDeath {
        initial: 4,
        birth_rate: 1.0,
        death_rate: 0.25,
    };
    let mut rng = SmallRng::seed_from_u64(11);
    let mut direct = Simulation::direct(model)?;
    let mut dense = Simulation::dense_direct(DenseBirthDeath(model))?;

    assert_eq!(direct.scheduler().stats().active_channel_count, 2);
    assert_eq!(dense.scheduler().stats().active_channel_count, 2);

    assert_eq!(direct.run(&mut rng, 16)?, RunStatus::MaxEvents {
        fired_events: 16
    });
    assert_eq!(dense.run(&mut rng, 16)?, RunStatus::MaxEvents {
        fired_events: 16
    });
    assert!(direct.time().is_finite());
    assert!(dense.time().is_finite());
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct StaticBirthDeath {
    initial: u64,
    birth_rate: f64,
    death_rate: f64,
}

#[derive(Debug, Clone, Copy)]
struct Birth;

#[derive(Debug, Clone, Copy)]
struct Death;

impl StaticDirectModel for StaticBirthDeath {
    type State = u64;

    fn initial_state(&self) -> Self::State {
        self.initial
    }
}

impl StaticReactionFamily<StaticBirthDeath> for Birth {
    fn channel_count(&self, _model: &StaticBirthDeath, _state: &u64) -> usize {
        1
    }

    fn propensity(
        &self,
        model: &StaticBirthDeath,
        state: &u64,
        _local_channel: usize,
        _time: f64,
    ) -> f64 {
        model.birth_rate * (*state as f64 + 1.0)
    }

    fn fire<R, U>(
        &self,
        _model: &StaticBirthDeath,
        state: &mut u64,
        _local_channel: usize,
        _rng: &mut R,
        updates: &mut U,
    ) where
        R: Rng + ?Sized,
        U: StaticUpdateSink + ?Sized,
    {
        *state += 1;
        updates.recompute_family(0, 0);
        updates.recompute_family(1, 0);
    }
}

impl StaticReactionFamily<StaticBirthDeath> for Death {
    fn channel_count(&self, _model: &StaticBirthDeath, _state: &u64) -> usize {
        1
    }

    fn propensity(
        &self,
        model: &StaticBirthDeath,
        state: &u64,
        _local_channel: usize,
        _time: f64,
    ) -> f64 {
        model.death_rate * *state as f64
    }

    fn fire<R, U>(
        &self,
        _model: &StaticBirthDeath,
        state: &mut u64,
        _local_channel: usize,
        _rng: &mut R,
        updates: &mut U,
    ) where
        R: Rng + ?Sized,
        U: StaticUpdateSink + ?Sized,
    {
        *state = state.saturating_sub(1);
        updates.recompute_family(0, 0);
        updates.recompute_family(1, 0);
    }
}

#[test]
fn static_family_schedulers_run_basic_birth_death_model() -> Result<()> {
    let model = StaticBirthDeath {
        initial: 4,
        birth_rate: 1.0,
        death_rate: 0.25,
    };

    let mut rng = SmallRng::seed_from_u64(19);
    let mut static_direct = StaticDirect::new(model, (Birth, Death))?;
    let mut family_direct = FamilyDirect::new(model, (Birth, Death))?;
    let mut family_nrm = FamilyNrm::new(model, (Birth, Death), &mut rng)?;
    let mut static_family_nrm = StaticFamilyNrm::<_, _, 4>::new(model, (Birth, Death), &mut rng)?;

    assert_eq!(static_direct.run(&mut rng, 16)?, 16);
    assert_eq!(family_direct.run(&mut rng, 16)?, 16);
    assert_eq!(family_nrm.run(&mut rng, 16)?, 16);
    assert_eq!(static_family_nrm.run(&mut rng, 16)?, 16);

    assert!(static_direct.time().is_finite());
    assert!(family_direct.time().is_finite());
    assert!(family_nrm.time().is_finite());
    assert!(static_family_nrm.time().is_finite());
    Ok(())
}
