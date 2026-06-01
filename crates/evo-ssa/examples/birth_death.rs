use evo_ssa::prelude::*;
use rand::{SeedableRng, rngs::SmallRng};

#[derive(Debug, Clone)]
struct BirthDeath {
    initial_count: u64,
    birth_rate: f64,
    death_rate: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Channel {
    Birth,
    Death,
}

#[derive(Debug, Clone, Copy)]
enum Event {
    Birth,
    Death,
}

impl EvolvingModel for BirthDeath {
    type ChannelKey = Channel;
    type ChannelPayload = ();
    type Event = Event;
    type State = u64;

    fn initial_state(&self) -> Self::State {
        self.initial_count
    }

    fn initialize_channels(
        &self,
        _state: &Self::State,
        channels: &mut ChannelEditor<Self::ChannelKey, Self::ChannelPayload>,
    ) {
        channels.upsert(Channel::Birth, ());
        channels.upsert(Channel::Death, ());
    }

    fn propensity(
        &self,
        state: &Self::State,
        key: Self::ChannelKey,
        _payload: &Self::ChannelPayload,
        _time: f64,
    ) -> f64 {
        match key {
            Channel::Birth => self.birth_rate * *state as f64,
            Channel::Death => self.death_rate * *state as f64,
        }
    }

    fn fire<R: rand::Rng + ?Sized>(
        &self,
        state: &mut Self::State,
        key: Self::ChannelKey,
        _payload: &Self::ChannelPayload,
        _rng: &mut R,
    ) -> Self::Event {
        match key {
            Channel::Birth => {
                *state += 1;
                Event::Birth
            }
            Channel::Death => {
                *state = state.saturating_sub(1);
                Event::Death
            }
        }
    }

    fn refresh_after_event(
        &self,
        _state: &Self::State,
        _event: &Self::Event,
        channels: &mut ChannelEditor<Self::ChannelKey, Self::ChannelPayload>,
    ) {
        channels.recompute(Channel::Birth);
        channels.recompute(Channel::Death);
    }
}

fn main() -> evo_ssa::Result<()> {
    let model = BirthDeath {
        initial_count: 100,
        birth_rate: 1.0,
        death_rate: 0.25,
    };
    let mut simulation = Simulation::direct(model)?;
    let mut rng = SmallRng::seed_from_u64(42);
    let status = simulation.run(&mut rng, 1_000)?;

    println!(
        "status={status:?}, time={:.4}, final_count={}",
        simulation.time(),
        simulation.state()
    );
    Ok(())
}
