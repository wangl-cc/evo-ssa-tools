use evo_ssa::prelude::*;
use rand::{Rng, RngExt, SeedableRng, rngs::SmallRng};

fn rngs(seed: u64) -> SsaRngs<SmallRng, SmallRng, SmallRng> {
    SsaRngs::new(
        SmallRng::seed_from_u64(seed),
        SmallRng::seed_from_u64(seed + 1),
        SmallRng::seed_from_u64(seed + 2),
    )
}

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

#[test]
fn dynamic_direct_runs_basic_birth_death_model() -> Result<()> {
    let model = DynamicBirthDeath {
        initial: 4,
        birth_rate: 1.0,
        death_rate: 0.25,
    };
    let mut simulation = Simulation::direct(model)?;
    let mut random = rngs(11);

    assert_eq!(simulation.scheduler().stats().active_channel_count, 2);
    assert_eq!(simulation.run(&mut random, 16)?, RunStatus::MaxEvents {
        fired_events: 16
    });
    assert!(simulation.time().is_finite());
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

reaction_families! {
    type BirthDeathFamilies;
    enum BirthDeathFamily {
        Birth => Birth,
        Death => Death,
    }
}

#[derive(Debug, Clone, Copy)]
struct PopulationChanged;

impl StaticFamilyModel for StaticBirthDeath {
    type Event = PopulationChanged;
    type Families = BirthDeathFamilies;
    type State = u64;

    fn initial_state(&self) -> Self::State {
        self.initial
    }

    fn refresh_after_event<U>(&self, _state: &u64, _event: &PopulationChanged, updates: &mut U)
    where
        U: ChannelRecomputeSink<Self::Families> + ?Sized,
    {
        updates.recompute_family(BirthDeathFamily::Birth, 0);
        updates.recompute_family(BirthDeathFamily::Death, 0);
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

    fn fire<R: Rng + ?Sized>(
        &self,
        _model: &StaticBirthDeath,
        state: &mut u64,
        _local_channel: usize,
        _rng: &mut R,
    ) -> PopulationChanged {
        *state += 1;
        PopulationChanged
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

    fn fire<R: Rng + ?Sized>(
        &self,
        _model: &StaticBirthDeath,
        state: &mut u64,
        _local_channel: usize,
        _rng: &mut R,
    ) -> PopulationChanged {
        *state = state.saturating_sub(1);
        PopulationChanged
    }
}

#[test]
fn family_schedulers_share_the_same_public_simulation_lifecycle() -> Result<()> {
    let model = StaticBirthDeath {
        initial: 4,
        birth_rate: 1.0,
        death_rate: 0.25,
    };

    let mut direct_random = rngs(19);
    let mut family_direct = FamilyDirect::new(model, family_list![Birth, Death])?;
    assert_eq!(
        family_direct.run(&mut direct_random, 16)?,
        RunStatus::MaxEvents { fired_events: 16 }
    );

    let mut nrm_random = rngs(29);
    let mut family_nrm = FamilyNrm::new(model, family_list![Birth, Death], nrm_random.clock_mut())?;
    assert_eq!(family_nrm.run(&mut nrm_random, 16)?, RunStatus::MaxEvents {
        fired_events: 16
    });

    let mut static_random = rngs(39);
    let mut static_nrm = StaticFamilyNrm::<_, _, 1>::new(
        model,
        family_list![Birth, Death],
        static_random.clock_mut(),
    )?;
    assert_eq!(
        static_nrm.run(&mut static_random, 16)?,
        RunStatus::MaxEvents { fired_events: 16 }
    );

    assert!(family_direct.time().is_finite());
    assert!(family_nrm.time().is_finite());
    assert!(static_nrm.time().is_finite());
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct ChannelGrowthModel;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ChannelGrowthState {
    stage: u8,
}

#[derive(Debug, Clone, Copy)]
struct BeforeGrowth;

#[derive(Debug, Clone, Copy)]
struct AfterGrowth;

reaction_families! {
    type GrowthFamilies;
    enum GrowthFamily {
        Before => BeforeGrowth,
        After => AfterGrowth,
    }
}

impl StaticFamilyModel for ChannelGrowthModel {
    type Event = ();
    type Families = GrowthFamilies;
    type State = ChannelGrowthState;

    fn initial_state(&self) -> Self::State {
        ChannelGrowthState { stage: 0 }
    }

    fn refresh_after_event<U>(&self, state: &ChannelGrowthState, _event: &(), updates: &mut U)
    where
        U: ChannelRecomputeSink<Self::Families> + ?Sized,
    {
        updates.recompute_family(GrowthFamily::Before, 0);
        updates.recompute_family(GrowthFamily::After, 0);
        if state.stage == 1 {
            updates.recompute_family(GrowthFamily::Before, 1);
            updates.recompute_family(GrowthFamily::After, 1);
        }
    }
}

impl StaticReactionFamily<ChannelGrowthModel> for BeforeGrowth {
    fn channel_count(&self, _model: &ChannelGrowthModel, state: &ChannelGrowthState) -> usize {
        if state.stage == 1 { 2 } else { 1 }
    }

    fn propensity(
        &self,
        _model: &ChannelGrowthModel,
        state: &ChannelGrowthState,
        local_channel: usize,
        _time: f64,
    ) -> f64 {
        if matches!(state.stage, 0 | 2) && local_channel == 0 {
            1.0
        } else {
            0.0
        }
    }

    fn fire<R: Rng + ?Sized>(
        &self,
        _model: &ChannelGrowthModel,
        state: &mut ChannelGrowthState,
        _local_channel: usize,
        _rng: &mut R,
    ) {
        state.stage += 1;
    }
}

impl StaticReactionFamily<ChannelGrowthModel> for AfterGrowth {
    fn channel_count(&self, _model: &ChannelGrowthModel, state: &ChannelGrowthState) -> usize {
        if state.stage == 1 { 2 } else { 1 }
    }

    fn propensity(
        &self,
        _model: &ChannelGrowthModel,
        state: &ChannelGrowthState,
        local_channel: usize,
        _time: f64,
    ) -> f64 {
        if state.stage == 1 && local_channel == 1 {
            1.0
        } else {
            0.0
        }
    }

    fn fire<R: Rng + ?Sized>(
        &self,
        _model: &ChannelGrowthModel,
        state: &mut ChannelGrowthState,
        _local_channel: usize,
        _rng: &mut R,
    ) {
        state.stage += 1;
    }
}

#[test]
fn family_schedulers_resize_channels_after_growth_and_shrink() -> Result<()> {
    let mut direct_random = rngs(49);
    let mut direct =
        FamilyDirect::new(ChannelGrowthModel, family_list![BeforeGrowth, AfterGrowth])?;
    for expected_stage in 1..=3 {
        assert!(direct.step(&mut direct_random)?.is_some());
        assert_eq!(direct.state().stage, expected_stage);
    }

    let mut nrm_random = rngs(59);
    let mut nrm = FamilyNrm::new(
        ChannelGrowthModel,
        family_list![BeforeGrowth, AfterGrowth],
        nrm_random.clock_mut(),
    )?;
    for expected_stage in 1..=3 {
        assert!(nrm.step(&mut nrm_random)?.is_some());
        assert_eq!(nrm.state().stage, expected_stage);
    }

    let mut static_random = rngs(69);
    let mut static_nrm = StaticFamilyNrm::<_, _, 1>::new(
        ChannelGrowthModel,
        family_list![BeforeGrowth, AfterGrowth],
        static_random.clock_mut(),
    )?;
    for expected_stage in 1..=3 {
        assert!(static_nrm.step(&mut static_random)?.is_some());
        assert_eq!(static_nrm.state().stage, expected_stage);
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct InvalidUpdateModel;

#[derive(Debug, Clone, Copy)]
struct InvalidUpdateFamily;

reaction_families! {
    type InvalidUpdateFamilies;
    enum InvalidUpdateFamilyId {
        Only => InvalidUpdateFamily,
    }
}

impl StaticFamilyModel for InvalidUpdateModel {
    type Event = ();
    type Families = InvalidUpdateFamilies;
    type State = ();

    fn initial_state(&self) -> Self::State {}

    fn refresh_after_event<U>(&self, _state: &(), _event: &(), updates: &mut U)
    where
        U: ChannelRecomputeSink<Self::Families> + ?Sized,
    {
        updates.recompute_family(InvalidUpdateFamilyId::Only, 1);
    }
}

impl StaticReactionFamily<InvalidUpdateModel> for InvalidUpdateFamily {
    fn channel_count(&self, _model: &InvalidUpdateModel, _state: &()) -> usize {
        1
    }

    fn propensity(
        &self,
        _model: &InvalidUpdateModel,
        _state: &(),
        _local_channel: usize,
        _time: f64,
    ) -> f64 {
        1.0
    }

    fn fire<R: Rng + ?Sized>(
        &self,
        _model: &InvalidUpdateModel,
        _state: &mut (),
        _local_channel: usize,
        _rng: &mut R,
    ) {
    }
}

#[test]
fn invalid_family_updates_fail_closed_and_poison_the_simulation() -> Result<()> {
    let mut simulation = FamilyDirect::new(InvalidUpdateModel, family_list![InvalidUpdateFamily])?;
    let mut random = rngs(79);

    assert!(matches!(
        simulation.step(&mut random),
        Err(Error::InvalidFamilyChannel {
            family: 0,
            local_channel: 1,
            channel_count: 1,
        })
    ));
    assert!(matches!(
        simulation.step(&mut random),
        Err(Error::SimulationPoisoned)
    ));
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct ConstantFamily {
    event: u8,
    propensity: f64,
}

#[derive(Debug, Clone, Copy)]
struct ManyFamilyModel;

reaction_families! {
    type ManyFamilies;
    enum ManyFamilyId {
        F0 => ConstantFamily,
        F1 => ConstantFamily,
        F2 => ConstantFamily,
        F3 => ConstantFamily,
        F4 => ConstantFamily,
        F5 => ConstantFamily,
        F6 => ConstantFamily,
        F7 => ConstantFamily,
        F8 => ConstantFamily,
        F9 => ConstantFamily,
        F10 => ConstantFamily,
        F11 => ConstantFamily,
        F12 => ConstantFamily,
    }
}

impl StaticFamilyModel for ManyFamilyModel {
    type Event = u8;
    type Families = ManyFamilies;
    type State = ();

    fn initial_state(&self) -> Self::State {}

    fn refresh_after_event<U>(&self, _state: &(), _event: &u8, _updates: &mut U)
    where
        U: ChannelRecomputeSink<Self::Families> + ?Sized,
    {
    }
}

impl StaticReactionFamily<ManyFamilyModel> for ConstantFamily {
    fn channel_count(&self, _model: &ManyFamilyModel, _state: &()) -> usize {
        1
    }

    fn propensity(
        &self,
        _model: &ManyFamilyModel,
        _state: &(),
        _local_channel: usize,
        _time: f64,
    ) -> f64 {
        self.propensity
    }

    fn fire<R: Rng + ?Sized>(
        &self,
        _model: &ManyFamilyModel,
        _state: &mut (),
        _local_channel: usize,
        _rng: &mut R,
    ) -> u8 {
        self.event
    }
}

fn many_families() -> ManyFamilies {
    family_list![
        ConstantFamily {
            event: 0,
            propensity: 0.0
        },
        ConstantFamily {
            event: 1,
            propensity: 0.0
        },
        ConstantFamily {
            event: 2,
            propensity: 0.0
        },
        ConstantFamily {
            event: 3,
            propensity: 0.0
        },
        ConstantFamily {
            event: 4,
            propensity: 0.0
        },
        ConstantFamily {
            event: 5,
            propensity: 0.0
        },
        ConstantFamily {
            event: 6,
            propensity: 0.0
        },
        ConstantFamily {
            event: 7,
            propensity: 0.0
        },
        ConstantFamily {
            event: 8,
            propensity: 0.0
        },
        ConstantFamily {
            event: 9,
            propensity: 0.0
        },
        ConstantFamily {
            event: 10,
            propensity: 0.0
        },
        ConstantFamily {
            event: 11,
            propensity: 0.0
        },
        ConstantFamily {
            event: 12,
            propensity: 1.0
        },
    ]
}

#[test]
fn static_family_lists_support_more_than_twelve_reaction_families() -> Result<()> {
    let all_ids: [FamilyId<ManyFamilies>; 13] = [
        ManyFamilyId::F0.into(),
        ManyFamilyId::F1.into(),
        ManyFamilyId::F2.into(),
        ManyFamilyId::F3.into(),
        ManyFamilyId::F4.into(),
        ManyFamilyId::F5.into(),
        ManyFamilyId::F6.into(),
        ManyFamilyId::F7.into(),
        ManyFamilyId::F8.into(),
        ManyFamilyId::F9.into(),
        ManyFamilyId::F10.into(),
        ManyFamilyId::F11.into(),
        ManyFamilyId::F12.into(),
    ];
    assert_eq!(all_ids[12].index(), 12);

    let mut direct_random = rngs(89);
    let mut direct = FamilyDirect::new(ManyFamilyModel, many_families())?;
    assert_eq!(direct.step(&mut direct_random)?, Some(12));

    let mut static_random = rngs(99);
    let mut static_nrm = StaticFamilyNrm::<_, _, 0>::new(
        ManyFamilyModel,
        many_families(),
        static_random.clock_mut(),
    )?;
    assert_eq!(static_nrm.step(&mut static_random)?, Some(12));
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct SlotModel;

impl EvolvingModel for SlotModel {
    type ChannelKey = u64;
    type ChannelPayload = ();
    type Event = ();
    type State = ();

    fn initial_state(&self) -> Self::State {}

    fn initialize_channels(
        &self,
        _state: &Self::State,
        _channels: &mut ChannelEditor<Self::ChannelKey, Self::ChannelPayload>,
    ) {
    }

    fn propensity(
        &self,
        _state: &Self::State,
        _key: Self::ChannelKey,
        _payload: &Self::ChannelPayload,
        _time: f64,
    ) -> f64 {
        1.0
    }

    fn fire<R: Rng + ?Sized>(
        &self,
        _state: &mut Self::State,
        _key: Self::ChannelKey,
        _payload: &Self::ChannelPayload,
        _rng: &mut R,
    ) -> Self::Event {
    }

    fn refresh_after_event(
        &self,
        _state: &Self::State,
        _event: &Self::Event,
        _channels: &mut ChannelEditor<Self::ChannelKey, Self::ChannelPayload>,
    ) {
    }
}

#[test]
fn dynamic_direct_reuses_removed_slots_and_rejects_missing_recomputes() -> Result<()> {
    let model = SlotModel;
    let mut scheduler = DirectScheduler::new();
    scheduler.initialize(&model, &(), 0.0, [ChannelUpdate::Upsert {
        key: 0,
        payload: (),
    }])?;

    for key in 1..100 {
        scheduler.apply_updates(&model, &(), 0.0, [
            ChannelUpdate::Remove { key: key - 1 },
            ChannelUpdate::Upsert { key, payload: () },
        ])?;
    }

    assert_eq!(scheduler.stats().active_channel_count, 1);
    assert_eq!(scheduler.stats().allocated_slot_count, 1);
    assert!(matches!(
        scheduler.apply_updates(&model, &(), 0.0, [ChannelUpdate::Recompute { key: 0 }]),
        Err(Error::MissingRecomputeChannel)
    ));
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct RandomEventModel;

impl EvolvingModel for RandomEventModel {
    type ChannelKey = ();
    type ChannelPayload = ();
    type Event = u64;
    type State = ();

    fn initial_state(&self) -> Self::State {}

    fn initialize_channels(
        &self,
        _state: &Self::State,
        channels: &mut ChannelEditor<Self::ChannelKey, Self::ChannelPayload>,
    ) {
        channels.upsert((), ());
    }

    fn propensity(
        &self,
        _state: &Self::State,
        _key: Self::ChannelKey,
        _payload: &Self::ChannelPayload,
        _time: f64,
    ) -> f64 {
        1.0
    }

    fn fire<R: Rng + ?Sized>(
        &self,
        _state: &mut Self::State,
        _key: Self::ChannelKey,
        _payload: &Self::ChannelPayload,
        rng: &mut R,
    ) -> Self::Event {
        rng.random()
    }

    fn refresh_after_event(
        &self,
        _state: &Self::State,
        _event: &Self::Event,
        channels: &mut ChannelEditor<Self::ChannelKey, Self::ChannelPayload>,
    ) {
        channels.recompute(());
    }
}

#[test]
fn event_randomness_is_independent_of_clock_and_selection_streams() -> Result<()> {
    let mut left = Simulation::direct(RandomEventModel)?;
    let mut right = Simulation::direct(RandomEventModel)?;
    let event_seed = 1234;
    let mut left_random = SsaRngs::new(
        SmallRng::seed_from_u64(1),
        SmallRng::seed_from_u64(2),
        SmallRng::seed_from_u64(event_seed),
    );
    let mut right_random = SsaRngs::new(
        SmallRng::seed_from_u64(101),
        SmallRng::seed_from_u64(202),
        SmallRng::seed_from_u64(event_seed),
    );

    for _ in 0..32 {
        assert_eq!(left.step(&mut left_random)?, right.step(&mut right_random)?);
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct SharedFamilyShape;

reaction_families! {
    type FirstBrand;
    enum FirstBrandId {
        Shared => SharedFamilyShape,
    }
}

reaction_families! {
    type SecondBrand;
    enum SecondBrandId {
        Shared => SharedFamilyShape,
    }
}

#[test]
fn structurally_identical_family_lists_keep_distinct_nominal_brands() {
    use std::any::TypeId;

    let _: FamilyId<FirstBrand> = FirstBrandId::Shared.into();
    let _: FamilyId<SecondBrand> = SecondBrandId::Shared.into();
    assert_ne!(TypeId::of::<FirstBrand>(), TypeId::of::<SecondBrand>());
}
