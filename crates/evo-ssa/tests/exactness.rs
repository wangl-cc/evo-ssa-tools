use evo_ssa::prelude::*;
use rand::{Rng, SeedableRng, rngs::SmallRng};

const SAMPLE_COUNT: usize = 40_000;

fn rngs(seed: u64) -> SsaRngs<SmallRng, SmallRng, SmallRng> {
    SsaRngs::new(
        SmallRng::seed_from_u64(seed),
        SmallRng::seed_from_u64(seed + 1),
        SmallRng::seed_from_u64(seed + 2),
    )
}

#[derive(Debug, Clone, Copy)]
struct ConstantRates {
    rates: [f64; 2],
}

#[derive(Debug, Clone, Copy)]
struct ConstantRateFamily;

reaction_families! {
    type ConstantRateFamilies;
    enum ConstantRateFamilyId {
        Reactions => ConstantRateFamily,
    }
}

impl StaticFamilyModel for ConstantRates {
    type Event = usize;
    type Families = ConstantRateFamilies;
    type State = ();

    fn initial_state(&self) -> Self::State {}

    fn refresh_after_event<U>(&self, _state: &(), _event: &usize, _updates: &mut U)
    where
        U: ChannelRecomputeSink<Self::Families> + ?Sized,
    {
    }
}

impl StaticReactionFamily<ConstantRates> for ConstantRateFamily {
    fn channel_count(&self, _model: &ConstantRates, _state: &()) -> usize {
        2
    }

    fn propensity(
        &self,
        model: &ConstantRates,
        _state: &(),
        local_channel: usize,
        _time: f64,
    ) -> f64 {
        model.rates[local_channel]
    }

    fn fire<R: Rng + ?Sized>(
        &self,
        _model: &ConstantRates,
        _state: &mut (),
        local_channel: usize,
        _rng: &mut R,
    ) -> usize {
        local_channel
    }
}

fn count_first_channel(
    mut step: impl FnMut() -> Result<Option<usize>>,
    sample_count: usize,
) -> Result<usize> {
    let mut first_channel_count = 0;
    for _ in 0..sample_count {
        let event = step()?.ok_or(Error::MissingSelectedChannel)?;
        first_channel_count += usize::from(event == 0);
    }
    Ok(first_channel_count)
}

fn assert_close(observed: f64, expected: f64, tolerance: f64) {
    assert!(
        (observed - expected).abs() <= tolerance,
        "observed {observed}, expected {expected} +/- {tolerance}"
    );
}

#[test]
fn family_schedulers_match_the_exponential_waiting_time_mean() -> Result<()> {
    let model = ConstantRates { rates: [2.0, 0.0] };

    let mut direct_random = rngs(101);
    let mut direct = FamilyDirect::new(model, family_list![ConstantRateFamily])?;
    count_first_channel(|| direct.step(&mut direct_random), SAMPLE_COUNT)?;
    assert_close(direct.time() / SAMPLE_COUNT as f64, 0.5, 0.015);

    let mut nrm_random = rngs(201);
    let mut nrm = FamilyNrm::new(
        model,
        family_list![ConstantRateFamily],
        nrm_random.clock_mut(),
    )?;
    count_first_channel(|| nrm.step(&mut nrm_random), SAMPLE_COUNT)?;
    assert_close(nrm.time() / SAMPLE_COUNT as f64, 0.5, 0.015);

    let mut static_random = rngs(301);
    let mut static_nrm = StaticFamilyNrm::<_, _, 0>::new(
        model,
        family_list![ConstantRateFamily],
        static_random.clock_mut(),
    )?;
    count_first_channel(|| static_nrm.step(&mut static_random), SAMPLE_COUNT)?;
    assert_close(static_nrm.time() / SAMPLE_COUNT as f64, 0.5, 0.015);
    Ok(())
}

#[test]
fn family_schedulers_select_channels_in_propensity_proportion() -> Result<()> {
    let model = ConstantRates { rates: [1.0, 3.0] };

    let mut direct_random = rngs(401);
    let mut direct = FamilyDirect::new(model, family_list![ConstantRateFamily])?;
    let direct_fraction = count_first_channel(|| direct.step(&mut direct_random), SAMPLE_COUNT)?
        as f64
        / SAMPLE_COUNT as f64;
    assert_close(direct_fraction, 0.25, 0.015);

    let mut nrm_random = rngs(501);
    let mut nrm = FamilyNrm::new(
        model,
        family_list![ConstantRateFamily],
        nrm_random.clock_mut(),
    )?;
    let nrm_fraction = count_first_channel(|| nrm.step(&mut nrm_random), SAMPLE_COUNT)? as f64
        / SAMPLE_COUNT as f64;
    assert_close(nrm_fraction, 0.25, 0.015);

    let mut static_random = rngs(601);
    let mut static_nrm = StaticFamilyNrm::<_, _, 0>::new(
        model,
        family_list![ConstantRateFamily],
        static_random.clock_mut(),
    )?;
    let static_fraction = count_first_channel(|| static_nrm.step(&mut static_random), SAMPLE_COUNT)?
        as f64
        / SAMPLE_COUNT as f64;
    assert_close(static_fraction, 0.25, 0.015);
    Ok(())
}

#[test]
fn generated_family_ids_are_bundle_branded_and_named() {
    let id: FamilyId<ConstantRateFamilies> = ConstantRateFamilyId::Reactions.into();
    assert_eq!(id.index(), 0);
}
