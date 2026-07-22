use evo_ssa::prelude::*;
use rand::{Rng, RngExt, SeedableRng, rngs::SmallRng};

const CHANNEL_COUNT: usize = 8;
const EVENT_COUNT: usize = 5_000;

#[derive(Debug, Clone, Copy)]
struct SparseRateModel;

#[derive(Debug, Clone, Copy)]
struct SparseRateFamily;

reaction_families! {
    type SparseRateFamilies;
    enum SparseRateFamilyId {
        Reactions => SparseRateFamily,
    }
}

impl StaticFamilyModel for SparseRateModel {
    type Event = usize;
    type Families = SparseRateFamilies;
    type State = [u64; CHANNEL_COUNT];

    fn initial_state(&self) -> Self::State {
        std::array::from_fn(|index| index as u64 + 1)
    }

    fn refresh_after_event<U>(&self, _state: &Self::State, event: &usize, updates: &mut U)
    where
        U: ChannelRecomputeSink<Self::Families> + ?Sized,
    {
        updates.recompute_family(SparseRateFamilyId::Reactions, *event);
    }
}

impl StaticReactionFamily<SparseRateModel> for SparseRateFamily {
    fn channel_count(&self, _model: &SparseRateModel, _state: &[u64; CHANNEL_COUNT]) -> usize {
        CHANNEL_COUNT
    }

    fn propensity(
        &self,
        _model: &SparseRateModel,
        state: &[u64; CHANNEL_COUNT],
        local_channel: usize,
        _time: f64,
    ) -> f64 {
        state[local_channel] as f64
    }

    fn fire<R: Rng + ?Sized>(
        &self,
        _model: &SparseRateModel,
        state: &mut [u64; CHANNEL_COUNT],
        local_channel: usize,
        _rng: &mut R,
    ) -> usize {
        update_rate(state, local_channel);
        local_channel
    }
}

fn update_rate(state: &mut [u64; CHANNEL_COUNT], channel: usize) {
    state[channel] = state[channel] % 11 + 1;
}

fn rngs(seed: u64) -> SsaRngs<SmallRng, SmallRng, SmallRng> {
    SsaRngs::new(
        SmallRng::seed_from_u64(seed),
        SmallRng::seed_from_u64(seed + 1),
        SmallRng::seed_from_u64(seed + 2),
    )
}

fn reference_step(
    state: &mut [u64; CHANNEL_COUNT],
    time: &mut f64,
    random: &mut SsaRngs<SmallRng, SmallRng, SmallRng>,
) -> usize {
    let total: f64 = state.iter().map(|&rate| rate as f64).sum();
    let waiting_draw = random
        .clock_mut()
        .random::<f64>()
        .clamp(f64::MIN_POSITIVE, 1.0);
    *time += -waiting_draw.ln() / total;

    let mut target = random.selection_mut().random::<f64>() * total;
    let mut selected = 0;
    for (channel, &rate) in state.iter().enumerate() {
        if target < rate as f64 {
            selected = channel;
            break;
        }
        target -= rate as f64;
    }
    update_rate(state, selected);
    selected
}

#[test]
fn cached_family_direct_matches_a_full_rebuild_reference() -> Result<()> {
    let mut simulation = FamilyDirect::new(SparseRateModel, family_list![SparseRateFamily])?;
    let mut framework_random = rngs(701);
    let mut reference_random = rngs(701);
    let mut reference_state = SparseRateModel.initial_state();
    let mut reference_time = 0.0;

    for _ in 0..EVENT_COUNT {
        let framework_event = simulation
            .step(&mut framework_random)?
            .ok_or(Error::MissingSelectedChannel)?;
        let reference_event = reference_step(
            &mut reference_state,
            &mut reference_time,
            &mut reference_random,
        );

        assert_eq!(framework_event, reference_event);
        assert_eq!(simulation.state(), &reference_state);
        assert_eq!(simulation.time(), reference_time);
    }
    Ok(())
}
