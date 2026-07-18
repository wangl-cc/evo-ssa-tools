use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use evo_ssa::prelude::*;
use rand::{Rng, RngExt, SeedableRng, rngs::SmallRng};

const ONE_SPECIES_EVENTS: usize = 20_000;
const MANY_SPECIES: usize = 4_096;
const MANY_SPECIES_EVENTS: usize = 10_000;
const MUTATION_MAX_SPECIES: usize = 4_096;
const MUTATION_EVENTS: usize = 10_000;
const TRAIT_MUTATION_EVENTS: usize = 10_000;

fn simulation_rngs(seed: u64) -> SsaRngs<SmallRng, SmallRng, SmallRng> {
    SsaRngs::new(
        SmallRng::seed_from_u64(seed),
        SmallRng::seed_from_u64(seed + 1),
        SmallRng::seed_from_u64(seed + 2),
    )
}

fn open01<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    rng.random::<f64>().clamp(f64::MIN_POSITIVE, 1.0)
}

#[derive(Debug, Clone, Default)]
struct HandwrittenSegmentTree {
    weights: Vec<f64>,
    tree: Vec<f64>,
    leaf_count: usize,
}

impl HandwrittenSegmentTree {
    fn total(&self) -> f64 {
        self.tree.get(1).copied().unwrap_or(0.0)
    }

    fn push(&mut self, weight: f64) {
        self.weights.push(weight);
        self.ensure_capacity(self.weights.len());
        self.set(self.weights.len() - 1, weight);
    }

    fn set(&mut self, index: usize, weight: f64) {
        self.ensure_capacity(index + 1);
        self.weights[index] = weight;
        let mut tree_index = self.leaf_count + index;
        self.tree[tree_index] = weight;
        while tree_index > 1 {
            tree_index /= 2;
            self.tree[tree_index] = self.tree[tree_index * 2] + self.tree[tree_index * 2 + 1];
        }
    }

    fn sample(&self, target: f64) -> usize {
        let mut target = target;
        let mut tree_index = 1;
        while tree_index < self.leaf_count {
            let left = tree_index * 2;
            if target < self.tree[left] {
                tree_index = left;
            } else {
                target -= self.tree[left];
                tree_index = left + 1;
            }
        }
        tree_index - self.leaf_count
    }

    fn ensure_capacity(&mut self, len: usize) {
        if len <= self.leaf_count {
            return;
        }
        self.leaf_count = len.next_power_of_two().max(1);
        self.tree = vec![0.0; self.leaf_count * 2];
        for (index, &weight) in self.weights.iter().enumerate() {
            self.tree[self.leaf_count + index] = weight;
        }
        for index in (1..self.leaf_count).rev() {
            self.tree[index] = self.tree[index * 2] + self.tree[index * 2 + 1];
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct HandwrittenClock {
    propensity: f64,
    scheduled_time: f64,
}

impl HandwrittenClock {
    fn inactive() -> Self {
        Self {
            propensity: 0.0,
            scheduled_time: f64::INFINITY,
        }
    }
}

#[derive(Debug, Clone)]
struct HandwrittenNrmClocks {
    clocks: Vec<HandwrittenClock>,
    heap: HandwrittenIndexedMinHeap,
}

impl HandwrittenNrmClocks {
    fn new(channel_count: usize) -> Self {
        Self {
            clocks: vec![HandwrittenClock::inactive(); channel_count],
            heap: HandwrittenIndexedMinHeap::with_capacity(channel_count),
        }
    }

    fn initialize<R: Rng + ?Sized>(
        &mut self,
        channel: usize,
        propensity: f64,
        now: f64,
        rng: &mut R,
    ) {
        let scheduled_time = draw_handwritten_scheduled_time(now, propensity, rng);
        self.clocks[channel] = HandwrittenClock {
            propensity,
            scheduled_time,
        };
        self.update_heap(channel, scheduled_time);
    }

    fn reschedule<R: Rng + ?Sized>(
        &mut self,
        channel: usize,
        new_propensity: f64,
        now: f64,
        fired: bool,
        rng: &mut R,
    ) {
        let old = self.clocks[channel];
        let scheduled_time = if new_propensity <= 0.0 {
            f64::INFINITY
        } else if fired || old.propensity <= 0.0 || !old.scheduled_time.is_finite() {
            draw_handwritten_scheduled_time(now, new_propensity, rng)
        } else {
            let residual = (old.scheduled_time - now).max(0.0);
            now + (old.propensity / new_propensity) * residual
        };

        self.clocks[channel] = HandwrittenClock {
            propensity: new_propensity,
            scheduled_time,
        };
        self.update_heap(channel, scheduled_time);
    }

    fn peek(&self) -> Option<(usize, f64)> {
        let channel = self.heap.peek()?;
        Some((channel, self.clocks[channel].scheduled_time))
    }

    fn update_heap(&mut self, channel: usize, scheduled_time: f64) {
        if scheduled_time.is_finite() {
            self.heap.insert_or_update(channel, scheduled_time);
        } else {
            self.heap.remove(channel);
        }
    }
}

#[derive(Debug, Clone)]
struct HandwrittenIndexedMinHeap {
    heap: Vec<usize>,
    positions: Vec<Option<usize>>,
    times: Vec<f64>,
}

impl HandwrittenIndexedMinHeap {
    fn with_capacity(item_count: usize) -> Self {
        Self {
            heap: Vec::new(),
            positions: vec![None; item_count],
            times: vec![f64::INFINITY; item_count],
        }
    }

    fn peek(&self) -> Option<usize> {
        self.heap.first().copied()
    }

    fn insert_or_update(&mut self, item: usize, time: f64) {
        let old_time = self.times[item];
        self.times[item] = time;

        if let Some(pos) = self.positions[item] {
            match time.total_cmp(&old_time) {
                std::cmp::Ordering::Less => self.sift_up(pos),
                std::cmp::Ordering::Greater => self.sift_down(pos),
                std::cmp::Ordering::Equal => {}
            }
        } else {
            let pos = self.heap.len();
            self.heap.push(item);
            self.positions[item] = Some(pos);
            self.sift_up(pos);
        }
    }

    fn remove(&mut self, item: usize) {
        let Some(pos) = self.positions[item] else {
            return;
        };

        let last = self.heap.pop().expect("position points into heap");
        self.positions[item] = None;
        if pos == self.heap.len() {
            return;
        }

        self.heap[pos] = last;
        self.positions[last] = Some(pos);
        self.sift_up(pos);
        let pos = self.positions[last].expect("heap position exists after sift up");
        self.sift_down(pos);
    }

    fn sift_up(&mut self, mut pos: usize) {
        while pos > 0 {
            let parent = (pos - 1) / 2;
            if self.less_or_equal(parent, pos) {
                break;
            }
            self.swap_positions(parent, pos);
            pos = parent;
        }
    }

    fn sift_down(&mut self, mut pos: usize) {
        loop {
            let left = pos * 2 + 1;
            let right = left + 1;
            let mut smallest = pos;

            if left < self.heap.len() && !self.less_or_equal(smallest, left) {
                smallest = left;
            }
            if right < self.heap.len() && !self.less_or_equal(smallest, right) {
                smallest = right;
            }
            if smallest == pos {
                break;
            }

            self.swap_positions(pos, smallest);
            pos = smallest;
        }
    }

    fn less_or_equal(&self, a_pos: usize, b_pos: usize) -> bool {
        let a = self.heap[a_pos];
        let b = self.heap[b_pos];
        self.times[a].total_cmp(&self.times[b]).is_le()
    }

    fn swap_positions(&mut self, a: usize, b: usize) {
        self.heap.swap(a, b);
        self.positions[self.heap[a]] = Some(a);
        self.positions[self.heap[b]] = Some(b);
    }
}

fn draw_handwritten_scheduled_time<R: Rng + ?Sized>(now: f64, propensity: f64, rng: &mut R) -> f64 {
    if propensity <= 0.0 {
        f64::INFINITY
    } else {
        now - open01(rng).ln() / propensity
    }
}

fn handwritten_one_species_birth_death(events: usize) -> (u64, f64) {
    let mut rng = SmallRng::seed_from_u64(42);
    let mut count = 100u64;
    let birth_rate = 1.0;
    let death_rate = 0.25;
    let mut time = 0.0;

    for _ in 0..events {
        let birth = birth_rate * count as f64;
        let death = death_rate * count as f64;
        let total = birth + death;
        if total <= 0.0 {
            break;
        }
        time += -open01(&mut rng).ln() / total;
        if rng.random::<f64>() * total < birth {
            count += 1;
        } else {
            count = count.saturating_sub(1);
        }
    }

    (count, time)
}

#[derive(Debug, Clone)]
struct OneSpeciesBirthDeath {
    initial_count: u64,
    birth_rate: f64,
    death_rate: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum OneSpeciesChannel {
    Birth,
    Death,
}

impl EvolvingModel for OneSpeciesBirthDeath {
    type ChannelKey = OneSpeciesChannel;
    type ChannelPayload = ();
    type Event = ();
    type State = u64;

    fn initial_state(&self) -> Self::State {
        self.initial_count
    }

    fn initialize_channels(
        &self,
        _state: &Self::State,
        channels: &mut ChannelEditor<Self::ChannelKey, Self::ChannelPayload>,
    ) {
        channels.upsert(OneSpeciesChannel::Birth, ());
        channels.upsert(OneSpeciesChannel::Death, ());
    }

    fn propensity(
        &self,
        state: &Self::State,
        key: Self::ChannelKey,
        _payload: &Self::ChannelPayload,
        _time: f64,
    ) -> f64 {
        match key {
            OneSpeciesChannel::Birth => self.birth_rate * *state as f64,
            OneSpeciesChannel::Death => self.death_rate * *state as f64,
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
            OneSpeciesChannel::Birth => *state += 1,
            OneSpeciesChannel::Death => *state = state.saturating_sub(1),
        }
    }

    fn refresh_after_event(
        &self,
        _state: &Self::State,
        _event: &Self::Event,
        channels: &mut ChannelEditor<Self::ChannelKey, Self::ChannelPayload>,
    ) {
        channels.recompute(OneSpeciesChannel::Birth);
        channels.recompute(OneSpeciesChannel::Death);
    }
}

fn dynamic_direct_one_species_birth_death(events: usize) -> (u64, f64) {
    let model = OneSpeciesBirthDeath {
        initial_count: 100,
        birth_rate: 1.0,
        death_rate: 0.25,
    };
    let mut simulation = Simulation::direct(model).expect("model has valid propensities");
    let mut rngs = simulation_rngs(42);
    simulation
        .run(&mut rngs, events)
        .expect("model has valid updates");
    (*simulation.state(), simulation.time())
}

#[derive(Debug, Clone)]
struct StaticOneSpeciesBirthDeath {
    initial_count: u64,
    birth_rate: f64,
    death_rate: f64,
}

impl StaticFamilyModel for StaticOneSpeciesBirthDeath {
    type Event = usize;
    type Families = StaticBirthDeathFamilies;
    type State = u64;

    fn initial_state(&self) -> Self::State {
        self.initial_count
    }

    fn refresh_after_event<U>(&self, _state: &u64, _event: &usize, updates: &mut U)
    where
        U: ChannelRecomputeSink<Self::Families> + ?Sized,
    {
        updates.recompute_family(StaticBirthDeathFamily::Birth, 0);
        updates.recompute_family(StaticBirthDeathFamily::Death, 0);
    }
}

#[derive(Debug, Clone, Copy)]
struct StaticBirth;

#[derive(Debug, Clone, Copy)]
struct StaticDeath;

reaction_families! {
    type StaticBirthDeathFamilies;
    enum StaticBirthDeathFamily {
        Birth => StaticBirth,
        Death => StaticDeath,
    }
}

impl StaticReactionFamily<StaticOneSpeciesBirthDeath> for StaticBirth {
    fn channel_count(
        &self,
        _model: &StaticOneSpeciesBirthDeath,
        _state: &<StaticOneSpeciesBirthDeath as StaticFamilyModel>::State,
    ) -> usize {
        1
    }

    fn propensity(
        &self,
        model: &StaticOneSpeciesBirthDeath,
        state: &<StaticOneSpeciesBirthDeath as StaticFamilyModel>::State,
        _local_channel: usize,
        _time: f64,
    ) -> f64 {
        model.birth_rate * *state as f64
    }

    fn fire<R: Rng + ?Sized>(
        &self,
        _model: &StaticOneSpeciesBirthDeath,
        state: &mut <StaticOneSpeciesBirthDeath as StaticFamilyModel>::State,
        _local_channel: usize,
        _rng: &mut R,
    ) -> usize {
        *state += 1;
        0
    }
}

impl StaticReactionFamily<StaticOneSpeciesBirthDeath> for StaticDeath {
    fn channel_count(
        &self,
        _model: &StaticOneSpeciesBirthDeath,
        _state: &<StaticOneSpeciesBirthDeath as StaticFamilyModel>::State,
    ) -> usize {
        1
    }

    fn propensity(
        &self,
        model: &StaticOneSpeciesBirthDeath,
        state: &<StaticOneSpeciesBirthDeath as StaticFamilyModel>::State,
        _local_channel: usize,
        _time: f64,
    ) -> f64 {
        model.death_rate * *state as f64
    }

    fn fire<R: Rng + ?Sized>(
        &self,
        _model: &StaticOneSpeciesBirthDeath,
        state: &mut <StaticOneSpeciesBirthDeath as StaticFamilyModel>::State,
        _local_channel: usize,
        _rng: &mut R,
    ) -> usize {
        *state = state.saturating_sub(1);
        0
    }
}

fn family_direct_one_species_birth_death(events: usize) -> (u64, f64) {
    let model = StaticOneSpeciesBirthDeath {
        initial_count: 100,
        birth_rate: 1.0,
        death_rate: 0.25,
    };
    let mut simulation = FamilyDirect::new(model, family_list![StaticBirth, StaticDeath])
        .expect("model has valid propensities");
    let mut rngs = simulation_rngs(42);
    simulation
        .run(&mut rngs, events)
        .expect("model has valid updates");
    (*simulation.state(), simulation.time())
}

fn family_nrm_one_species_birth_death(events: usize) -> (u64, f64) {
    let model = StaticOneSpeciesBirthDeath {
        initial_count: 100,
        birth_rate: 1.0,
        death_rate: 0.25,
    };
    let mut rngs = simulation_rngs(42);
    let mut simulation = FamilyNrm::new(
        model,
        family_list![StaticBirth, StaticDeath],
        rngs.clock_mut(),
    )
    .expect("model has valid propensities");
    simulation
        .run(&mut rngs, events)
        .expect("model has valid updates");
    (*simulation.state(), simulation.time())
}

fn static_family_nrm_one_species_birth_death(events: usize) -> (u64, f64) {
    let model = StaticOneSpeciesBirthDeath {
        initial_count: 100,
        birth_rate: 1.0,
        death_rate: 0.25,
    };
    let mut rngs = simulation_rngs(42);
    let mut simulation = StaticFamilyNrm::<_, _, 4>::new(
        model,
        family_list![StaticBirth, StaticDeath],
        rngs.clock_mut(),
    )
    .expect("model has valid propensities");
    simulation
        .run(&mut rngs, events)
        .expect("model has valid updates");
    (*simulation.state(), simulation.time())
}

fn many_birth_slot(species: usize) -> usize {
    species * 2
}

fn many_death_slot(species: usize) -> usize {
    species * 2 + 1
}

fn handwritten_many_species_birth_death(species_count: usize, events: usize) -> (u64, f64) {
    let mut rng = SmallRng::seed_from_u64(42);
    let mut counts = vec![10u64; species_count];
    let birth_rate = 1.0;
    let death_rate = 0.25;
    let mut tree = HandwrittenSegmentTree::default();
    for &count in &counts {
        tree.push(birth_rate * count as f64);
        tree.push(death_rate * count as f64);
    }
    let mut time = 0.0;

    for _ in 0..events {
        let total = tree.total();
        if total <= 0.0 {
            break;
        }
        time += -open01(&mut rng).ln() / total;
        let slot = tree.sample(rng.random::<f64>() * total);
        let species = slot / 2;
        if slot % 2 == 0 {
            counts[species] += 1;
        } else {
            counts[species] = counts[species].saturating_sub(1);
        }
        let count = counts[species] as f64;
        tree.set(many_birth_slot(species), birth_rate * count);
        tree.set(many_death_slot(species), death_rate * count);
    }

    (counts.iter().sum(), time)
}

#[derive(Debug, Clone)]
struct ManySpeciesBirthDeath {
    species_count: usize,
    initial_count: u64,
    birth_rate: f64,
    death_rate: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ManySpeciesChannel {
    Birth(usize),
    Death(usize),
}

#[derive(Debug, Clone, Copy)]
struct SpeciesPayload {
    species: usize,
}

#[derive(Debug, Clone, Copy)]
struct SpeciesEvent {
    species: usize,
}

impl EvolvingModel for ManySpeciesBirthDeath {
    type ChannelKey = ManySpeciesChannel;
    type ChannelPayload = SpeciesPayload;
    type Event = SpeciesEvent;
    type State = Vec<u64>;

    fn initial_state(&self) -> Self::State {
        vec![self.initial_count; self.species_count]
    }

    fn initialize_channels(
        &self,
        state: &Self::State,
        channels: &mut ChannelEditor<Self::ChannelKey, Self::ChannelPayload>,
    ) {
        for species in 0..state.len() {
            let payload = SpeciesPayload { species };
            channels.upsert(ManySpeciesChannel::Birth(species), payload);
            channels.upsert(ManySpeciesChannel::Death(species), payload);
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
            ManySpeciesChannel::Birth(_) => self.birth_rate * count,
            ManySpeciesChannel::Death(_) => self.death_rate * count,
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
            ManySpeciesChannel::Birth(_) => state[payload.species] += 1,
            ManySpeciesChannel::Death(_) => {
                state[payload.species] = state[payload.species].saturating_sub(1);
            }
        }
        SpeciesEvent {
            species: payload.species,
        }
    }

    fn refresh_after_event(
        &self,
        _state: &Self::State,
        event: &Self::Event,
        channels: &mut ChannelEditor<Self::ChannelKey, Self::ChannelPayload>,
    ) {
        channels.recompute(ManySpeciesChannel::Birth(event.species));
        channels.recompute(ManySpeciesChannel::Death(event.species));
    }
}

fn dynamic_direct_many_species_birth_death(species_count: usize, events: usize) -> (u64, f64) {
    let model = ManySpeciesBirthDeath {
        species_count,
        initial_count: 10,
        birth_rate: 1.0,
        death_rate: 0.25,
    };
    let mut simulation = Simulation::direct(model).expect("model has valid propensities");
    let mut rngs = simulation_rngs(42);
    simulation
        .run(&mut rngs, events)
        .expect("model has valid updates");
    (simulation.state().iter().sum(), simulation.time())
}

#[derive(Debug, Clone)]
struct StaticManySpeciesBirthDeath {
    species_count: usize,
    initial_count: u64,
    birth_rate: f64,
    death_rate: f64,
}

impl StaticFamilyModel for StaticManySpeciesBirthDeath {
    type Event = usize;
    type Families = StaticBirthDeathFamilies;
    type State = Vec<u64>;

    fn initial_state(&self) -> Self::State {
        vec![self.initial_count; self.species_count]
    }

    fn refresh_after_event<U>(&self, _state: &Vec<u64>, event: &usize, updates: &mut U)
    where
        U: ChannelRecomputeSink<Self::Families> + ?Sized,
    {
        updates.recompute_family(StaticBirthDeathFamily::Birth, *event);
        updates.recompute_family(StaticBirthDeathFamily::Death, *event);
    }
}

impl StaticReactionFamily<StaticManySpeciesBirthDeath> for StaticBirth {
    fn channel_count(
        &self,
        model: &StaticManySpeciesBirthDeath,
        _state: &<StaticManySpeciesBirthDeath as StaticFamilyModel>::State,
    ) -> usize {
        model.species_count
    }

    fn propensity(
        &self,
        model: &StaticManySpeciesBirthDeath,
        state: &<StaticManySpeciesBirthDeath as StaticFamilyModel>::State,
        local_channel: usize,
        _time: f64,
    ) -> f64 {
        model.birth_rate * state[local_channel] as f64
    }

    fn fire<R: Rng + ?Sized>(
        &self,
        _model: &StaticManySpeciesBirthDeath,
        state: &mut <StaticManySpeciesBirthDeath as StaticFamilyModel>::State,
        local_channel: usize,
        _rng: &mut R,
    ) -> usize {
        state[local_channel] += 1;
        local_channel
    }
}

impl StaticReactionFamily<StaticManySpeciesBirthDeath> for StaticDeath {
    fn channel_count(
        &self,
        model: &StaticManySpeciesBirthDeath,
        _state: &<StaticManySpeciesBirthDeath as StaticFamilyModel>::State,
    ) -> usize {
        model.species_count
    }

    fn propensity(
        &self,
        model: &StaticManySpeciesBirthDeath,
        state: &<StaticManySpeciesBirthDeath as StaticFamilyModel>::State,
        local_channel: usize,
        _time: f64,
    ) -> f64 {
        model.death_rate * state[local_channel] as f64
    }

    fn fire<R: Rng + ?Sized>(
        &self,
        _model: &StaticManySpeciesBirthDeath,
        state: &mut <StaticManySpeciesBirthDeath as StaticFamilyModel>::State,
        local_channel: usize,
        _rng: &mut R,
    ) -> usize {
        state[local_channel] = state[local_channel].saturating_sub(1);
        local_channel
    }
}

fn family_direct_many_species_birth_death(species_count: usize, events: usize) -> (u64, f64) {
    let model = StaticManySpeciesBirthDeath {
        species_count,
        initial_count: 10,
        birth_rate: 1.0,
        death_rate: 0.25,
    };
    let mut simulation = FamilyDirect::new(model, family_list![StaticBirth, StaticDeath])
        .expect("model has valid propensities");
    let mut rngs = simulation_rngs(42);
    simulation
        .run(&mut rngs, events)
        .expect("model has valid updates");
    (simulation.state().iter().sum(), simulation.time())
}

fn family_nrm_many_species_birth_death(species_count: usize, events: usize) -> (u64, f64) {
    let model = StaticManySpeciesBirthDeath {
        species_count,
        initial_count: 10,
        birth_rate: 1.0,
        death_rate: 0.25,
    };
    let mut rngs = simulation_rngs(42);
    let mut simulation = FamilyNrm::new(
        model,
        family_list![StaticBirth, StaticDeath],
        rngs.clock_mut(),
    )
    .expect("model has valid propensities");
    simulation
        .run(&mut rngs, events)
        .expect("model has valid updates");
    (simulation.state().iter().sum(), simulation.time())
}

fn static_family_nrm_many_species_birth_death(species_count: usize, events: usize) -> (u64, f64) {
    let model = StaticManySpeciesBirthDeath {
        species_count,
        initial_count: 10,
        birth_rate: 1.0,
        death_rate: 0.25,
    };
    let mut rngs = simulation_rngs(42);
    let mut simulation = StaticFamilyNrm::<_, _, 4>::new(
        model,
        family_list![StaticBirth, StaticDeath],
        rngs.clock_mut(),
    )
    .expect("model has valid propensities");
    simulation
        .run(&mut rngs, events)
        .expect("model has valid updates");
    (simulation.state().iter().sum(), simulation.time())
}

fn mutation_slot(species: usize) -> usize {
    species
}

fn handwritten_mutation_new_species(max_species: usize, events: usize) -> (usize, u64, f64) {
    let mut rng = SmallRng::seed_from_u64(42);
    let mut counts = vec![1_000u64];
    let mutation_rate = 1.0;
    let mut tree = HandwrittenSegmentTree::default();
    tree.push(mutation_rate * counts[0] as f64);
    let mut time = 0.0;

    for _ in 0..events {
        let total = tree.total();
        if total <= 0.0 {
            break;
        }
        time += -open01(&mut rng).ln() / total;
        let source = tree.sample(rng.random::<f64>() * total);
        counts[source] = counts[source].saturating_sub(1);
        let child = source + 1;
        if child < max_species {
            if child == counts.len() {
                counts.push(0);
                tree.push(0.0);
            }
            counts[child] += 1;
        }

        tree.set(mutation_slot(source), mutation_rate * counts[source] as f64);
        if child < counts.len() {
            tree.set(mutation_slot(child), mutation_rate * counts[child] as f64);
        }
    }

    (counts.len(), counts.iter().sum(), time)
}

#[derive(Debug, Clone)]
struct MutationNewSpecies {
    initial_count: u64,
    max_species: usize,
    mutation_rate: f64,
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

impl EvolvingModel for MutationNewSpecies {
    type ChannelKey = MutationChannel;
    type ChannelPayload = MutationPayload;
    type Event = MutationEvent;
    type State = Vec<u64>;

    fn initial_state(&self) -> Self::State {
        vec![self.initial_count]
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
        self.mutation_rate * state[payload.source] as f64
    }

    fn fire<R: Rng + ?Sized>(
        &self,
        state: &mut Self::State,
        _key: Self::ChannelKey,
        payload: &Self::ChannelPayload,
        _rng: &mut R,
    ) -> Self::Event {
        state[payload.source] = state[payload.source].saturating_sub(1);
        let child = payload.source + 1;
        let child = (child < self.max_species).then_some(child);
        if let Some(child) = child {
            if child == state.len() {
                state.push(0);
            }
            state[child] += 1;
        }
        MutationEvent {
            source: payload.source,
            child,
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

fn dynamic_direct_mutation_new_species(max_species: usize, events: usize) -> (usize, u64, f64) {
    let model = MutationNewSpecies {
        initial_count: 1_000,
        max_species,
        mutation_rate: 1.0,
    };
    let mut simulation = Simulation::direct(model).expect("model has valid propensities");
    let mut rngs = simulation_rngs(42);
    simulation
        .run(&mut rngs, events)
        .expect("model has valid updates");
    (
        simulation.state().len(),
        simulation.state().iter().sum(),
        simulation.time(),
    )
}

#[derive(Debug, Clone)]
struct StaticMutationNewSpecies {
    initial_count: u64,
    max_species: usize,
    mutation_rate: f64,
}

#[derive(Debug, Clone, Copy)]
struct StaticMutationEvent {
    source: usize,
    child: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
struct StaticMutation;

reaction_families! {
    type StaticMutationFamilies;
    enum StaticMutationFamily {
        Mutation => StaticMutation,
    }
}

impl StaticFamilyModel for StaticMutationNewSpecies {
    type Event = StaticMutationEvent;
    type Families = StaticMutationFamilies;
    type State = Vec<u64>;

    fn initial_state(&self) -> Self::State {
        vec![self.initial_count]
    }

    fn refresh_after_event<U>(
        &self,
        _state: &Vec<u64>,
        event: &StaticMutationEvent,
        updates: &mut U,
    ) where
        U: ChannelRecomputeSink<Self::Families> + ?Sized,
    {
        updates.recompute_family(StaticMutationFamily::Mutation, event.source);
        if let Some(child) = event.child {
            updates.recompute_family(StaticMutationFamily::Mutation, child);
        }
    }
}

impl StaticReactionFamily<StaticMutationNewSpecies> for StaticMutation {
    fn channel_count(
        &self,
        _model: &StaticMutationNewSpecies,
        state: &<StaticMutationNewSpecies as StaticFamilyModel>::State,
    ) -> usize {
        state.len()
    }

    fn propensity(
        &self,
        model: &StaticMutationNewSpecies,
        state: &<StaticMutationNewSpecies as StaticFamilyModel>::State,
        local_channel: usize,
        _time: f64,
    ) -> f64 {
        model.mutation_rate * state.get(local_channel).copied().unwrap_or(0) as f64
    }

    fn fire<R: Rng + ?Sized>(
        &self,
        model: &StaticMutationNewSpecies,
        state: &mut <StaticMutationNewSpecies as StaticFamilyModel>::State,
        local_channel: usize,
        _rng: &mut R,
    ) -> StaticMutationEvent {
        state[local_channel] = state[local_channel].saturating_sub(1);

        let child = local_channel + 1;
        let child = (child < model.max_species).then_some(child);
        if let Some(child) = child {
            if child == state.len() {
                state.push(0);
            }
            state[child] += 1;
        }
        StaticMutationEvent {
            source: local_channel,
            child,
        }
    }
}

fn family_direct_mutation_new_species(max_species: usize, events: usize) -> (usize, u64, f64) {
    let model = StaticMutationNewSpecies {
        initial_count: 1_000,
        max_species,
        mutation_rate: 1.0,
    };
    let mut simulation = FamilyDirect::new(model, family_list![StaticMutation])
        .expect("model has valid propensities");
    let mut rngs = simulation_rngs(42);
    simulation
        .run(&mut rngs, events)
        .expect("model has valid updates");
    (
        simulation.state().len(),
        simulation.state().iter().sum(),
        simulation.time(),
    )
}

fn family_nrm_mutation_new_species(max_species: usize, events: usize) -> (usize, u64, f64) {
    let model = StaticMutationNewSpecies {
        initial_count: 1_000,
        max_species,
        mutation_rate: 1.0,
    };
    let mut rngs = simulation_rngs(42);
    let mut simulation = FamilyNrm::new(model, family_list![StaticMutation], rngs.clock_mut())
        .expect("model has valid propensities");
    simulation
        .run(&mut rngs, events)
        .expect("model has valid updates");
    (
        simulation.state().len(),
        simulation.state().iter().sum(),
        simulation.time(),
    )
}

fn static_family_nrm_mutation_new_species(max_species: usize, events: usize) -> (usize, u64, f64) {
    let model = StaticMutationNewSpecies {
        initial_count: 1_000,
        max_species,
        mutation_rate: 1.0,
    };
    let mut rngs = simulation_rngs(42);
    let mut simulation =
        StaticFamilyNrm::<_, _, 4>::new(model, family_list![StaticMutation], rngs.clock_mut())
            .expect("model has valid propensities");
    simulation
        .run(&mut rngs, events)
        .expect("model has valid updates");
    (
        simulation.state().len(),
        simulation.state().iter().sum(),
        simulation.time(),
    )
}

#[derive(Debug, Clone, Copy)]
struct TraitSpecies {
    count: u64,
    birth_rate: f64,
    death_rate: f64,
    mutation_rate: f64,
}

fn mutate_rate<R: Rng + ?Sized>(rate: f64, rng: &mut R) -> f64 {
    let factor = 0.75 + 0.5 * rng.random::<f64>();
    (rate * factor).clamp(1.0e-6, 10.0)
}

fn update_trait_species_trees(
    species: &[TraitSpecies],
    max_species: usize,
    birth_tree: &mut HandwrittenSegmentTree,
    death_tree: &mut HandwrittenSegmentTree,
    mutation_tree: &mut HandwrittenSegmentTree,
    index: usize,
) {
    if index >= max_species {
        return;
    }
    let Some(sp) = species.get(index) else {
        birth_tree.set(index, 0.0);
        death_tree.set(index, 0.0);
        mutation_tree.set(index, 0.0);
        return;
    };
    birth_tree.set(index, sp.birth_rate * sp.count as f64);
    death_tree.set(index, sp.death_rate * sp.count as f64);
    mutation_tree.set(index, sp.mutation_rate * sp.count as f64);
}

fn handwritten_trait_mutation_birth_death(max_species: usize, events: usize) -> (usize, u64, f64) {
    let mut rngs = simulation_rngs(42);
    let mut species = vec![TraitSpecies {
        count: 1_000,
        birth_rate: 1.0,
        death_rate: 0.25,
        mutation_rate: 0.1,
    }];
    let mut birth_tree = HandwrittenSegmentTree::default();
    let mut death_tree = HandwrittenSegmentTree::default();
    let mut mutation_tree = HandwrittenSegmentTree::default();
    for index in 0..max_species {
        if index == 0 {
            birth_tree.push(species[0].birth_rate * species[0].count as f64);
            death_tree.push(species[0].death_rate * species[0].count as f64);
            mutation_tree.push(species[0].mutation_rate * species[0].count as f64);
        } else {
            birth_tree.push(0.0);
            death_tree.push(0.0);
            mutation_tree.push(0.0);
        }
    }

    let mut time = 0.0;
    for _ in 0..events {
        let birth_total = birth_tree.total();
        let death_total = death_tree.total();
        let mutation_total = mutation_tree.total();
        let total = birth_total + death_total + mutation_total;
        if total <= 0.0 {
            break;
        }

        time += -open01(rngs.clock_mut()).ln() / total;
        let family_draw = rngs.selection_mut().random::<f64>() * total;
        if family_draw < birth_total {
            let index = birth_tree.sample(rngs.selection_mut().random::<f64>() * birth_total);
            species[index].count += 1;
            update_trait_species_trees(
                &species,
                max_species,
                &mut birth_tree,
                &mut death_tree,
                &mut mutation_tree,
                index,
            );
        } else if family_draw < birth_total + death_total {
            let index = death_tree.sample(rngs.selection_mut().random::<f64>() * death_total);
            species[index].count = species[index].count.saturating_sub(1);
            update_trait_species_trees(
                &species,
                max_species,
                &mut birth_tree,
                &mut death_tree,
                &mut mutation_tree,
                index,
            );
        } else {
            let index = mutation_tree.sample(rngs.selection_mut().random::<f64>() * mutation_total);
            species[index].count = species[index].count.saturating_sub(1);
            let child = species.len();
            if child < max_species {
                let parent = species[index];
                species.push(TraitSpecies {
                    count: 1,
                    birth_rate: mutate_rate(parent.birth_rate, rngs.event_mut()),
                    death_rate: mutate_rate(parent.death_rate, rngs.event_mut()),
                    mutation_rate: mutate_rate(parent.mutation_rate, rngs.event_mut()),
                });
                update_trait_species_trees(
                    &species,
                    max_species,
                    &mut birth_tree,
                    &mut death_tree,
                    &mut mutation_tree,
                    child,
                );
            }
            update_trait_species_trees(
                &species,
                max_species,
                &mut birth_tree,
                &mut death_tree,
                &mut mutation_tree,
                index,
            );
        }
    }

    (species.len(), species.iter().map(|sp| sp.count).sum(), time)
}

const TRAIT_BIRTH_FAMILY: usize = 0;
const TRAIT_DEATH_FAMILY: usize = 1;
const TRAIT_MUTATION_FAMILY: usize = 2;

fn trait_nrm_propensity(species: &[TraitSpecies], family: usize, index: usize) -> f64 {
    let Some(sp) = species.get(index) else {
        return 0.0;
    };
    match family {
        TRAIT_BIRTH_FAMILY => sp.birth_rate * sp.count as f64,
        TRAIT_DEATH_FAMILY => sp.death_rate * sp.count as f64,
        TRAIT_MUTATION_FAMILY => sp.mutation_rate * sp.count as f64,
        _ => 0.0,
    }
}

fn update_trait_family_nrm_species(
    species: &[TraitSpecies],
    clocks: (
        &mut HandwrittenNrmClocks,
        &mut HandwrittenNrmClocks,
        &mut HandwrittenNrmClocks,
    ),
    index: usize,
    fired_family: Option<usize>,
    now: f64,
    rng: &mut impl Rng,
) {
    let (birth_clocks, death_clocks, mutation_clocks) = clocks;
    let Some(sp) = species.get(index) else {
        birth_clocks.reschedule(index, 0.0, now, false, rng);
        death_clocks.reschedule(index, 0.0, now, false, rng);
        mutation_clocks.reschedule(index, 0.0, now, false, rng);
        return;
    };

    birth_clocks.reschedule(
        index,
        sp.birth_rate * sp.count as f64,
        now,
        fired_family == Some(TRAIT_BIRTH_FAMILY),
        rng,
    );
    death_clocks.reschedule(
        index,
        sp.death_rate * sp.count as f64,
        now,
        fired_family == Some(TRAIT_DEATH_FAMILY),
        rng,
    );
    mutation_clocks.reschedule(
        index,
        sp.mutation_rate * sp.count as f64,
        now,
        fired_family == Some(TRAIT_MUTATION_FAMILY),
        rng,
    );
}

fn next_trait_family_nrm_channel(
    birth_clocks: &HandwrittenNrmClocks,
    death_clocks: &HandwrittenNrmClocks,
    mutation_clocks: &HandwrittenNrmClocks,
) -> Option<(usize, usize, f64)> {
    let mut selected = None;
    for (family, clocks) in [
        (TRAIT_BIRTH_FAMILY, birth_clocks),
        (TRAIT_DEATH_FAMILY, death_clocks),
        (TRAIT_MUTATION_FAMILY, mutation_clocks),
    ] {
        let Some((index, scheduled_time)) = clocks.peek() else {
            continue;
        };
        if selected.is_none_or(|(_, _, selected_time)| scheduled_time < selected_time) {
            selected = Some((family, index, scheduled_time));
        }
    }
    selected
}

fn handwritten_family_nrm_trait_mutation_birth_death(
    max_species: usize,
    events: usize,
) -> (usize, u64, f64) {
    let mut rngs = simulation_rngs(42);
    let mut species = vec![TraitSpecies {
        count: 1_000,
        birth_rate: 1.0,
        death_rate: 0.25,
        mutation_rate: 0.1,
    }];
    let mut birth_clocks = HandwrittenNrmClocks::new(max_species);
    let mut death_clocks = HandwrittenNrmClocks::new(max_species);
    let mut mutation_clocks = HandwrittenNrmClocks::new(max_species);
    let mut time = 0.0;

    for index in 0..max_species {
        let propensity = trait_nrm_propensity(&species, TRAIT_BIRTH_FAMILY, index);
        birth_clocks.initialize(index, propensity, time, rngs.clock_mut());
        let propensity = trait_nrm_propensity(&species, TRAIT_DEATH_FAMILY, index);
        death_clocks.initialize(index, propensity, time, rngs.clock_mut());
        let propensity = trait_nrm_propensity(&species, TRAIT_MUTATION_FAMILY, index);
        mutation_clocks.initialize(index, propensity, time, rngs.clock_mut());
    }

    for _ in 0..events {
        let Some((family, index, scheduled_time)) =
            next_trait_family_nrm_channel(&birth_clocks, &death_clocks, &mutation_clocks)
        else {
            break;
        };
        time = scheduled_time;

        match family {
            TRAIT_BIRTH_FAMILY => {
                species[index].count += 1;
                update_trait_family_nrm_species(
                    &species,
                    (&mut birth_clocks, &mut death_clocks, &mut mutation_clocks),
                    index,
                    Some(TRAIT_BIRTH_FAMILY),
                    time,
                    rngs.clock_mut(),
                );
            }
            TRAIT_DEATH_FAMILY => {
                species[index].count = species[index].count.saturating_sub(1);
                update_trait_family_nrm_species(
                    &species,
                    (&mut birth_clocks, &mut death_clocks, &mut mutation_clocks),
                    index,
                    Some(TRAIT_DEATH_FAMILY),
                    time,
                    rngs.clock_mut(),
                );
            }
            TRAIT_MUTATION_FAMILY => {
                let parent = species[index];
                species[index].count = species[index].count.saturating_sub(1);

                let child = species.len();
                if child < max_species {
                    species.push(TraitSpecies {
                        count: 1,
                        birth_rate: mutate_rate(parent.birth_rate, rngs.event_mut()),
                        death_rate: mutate_rate(parent.death_rate, rngs.event_mut()),
                        mutation_rate: mutate_rate(parent.mutation_rate, rngs.event_mut()),
                    });
                    update_trait_family_nrm_species(
                        &species,
                        (&mut birth_clocks, &mut death_clocks, &mut mutation_clocks),
                        child,
                        None,
                        time,
                        rngs.clock_mut(),
                    );
                }

                update_trait_family_nrm_species(
                    &species,
                    (&mut birth_clocks, &mut death_clocks, &mut mutation_clocks),
                    index,
                    Some(TRAIT_MUTATION_FAMILY),
                    time,
                    rngs.clock_mut(),
                );
            }
            _ => unreachable!("trait family NRM family is in range"),
        }
    }

    (species.len(), species.iter().map(|sp| sp.count).sum(), time)
}

#[derive(Debug, Clone)]
struct TraitMutationBirthDeath {
    initial_count: u64,
    max_species: usize,
    birth_rate: f64,
    death_rate: f64,
    mutation_rate: f64,
}

#[derive(Debug, Clone, Copy)]
struct TraitMutationEvent {
    parent: usize,
    child: Option<usize>,
}

reaction_families! {
    type TraitMutationFamilies;
    enum TraitMutationFamily {
        Birth => StaticBirth,
        Death => StaticDeath,
        Mutation => StaticMutation,
    }
}

impl StaticFamilyModel for TraitMutationBirthDeath {
    type Event = TraitMutationEvent;
    type Families = TraitMutationFamilies;
    type State = Vec<TraitSpecies>;

    fn initial_state(&self) -> Self::State {
        vec![TraitSpecies {
            count: self.initial_count,
            birth_rate: self.birth_rate,
            death_rate: self.death_rate,
            mutation_rate: self.mutation_rate,
        }]
    }

    fn refresh_after_event<U>(
        &self,
        _state: &Vec<TraitSpecies>,
        event: &TraitMutationEvent,
        updates: &mut U,
    ) where
        U: ChannelRecomputeSink<Self::Families> + ?Sized,
    {
        updates.recompute_family(TraitMutationFamily::Birth, event.parent);
        updates.recompute_family(TraitMutationFamily::Death, event.parent);
        updates.recompute_family(TraitMutationFamily::Mutation, event.parent);
        if let Some(child) = event.child {
            updates.reinitialize_family(TraitMutationFamily::Birth, child);
            updates.reinitialize_family(TraitMutationFamily::Death, child);
            updates.reinitialize_family(TraitMutationFamily::Mutation, child);
        }
    }
}

fn trait_model(max_species: usize) -> TraitMutationBirthDeath {
    TraitMutationBirthDeath {
        initial_count: 1_000,
        max_species,
        birth_rate: 1.0,
        death_rate: 0.25,
        mutation_rate: 0.1,
    }
}

impl StaticReactionFamily<TraitMutationBirthDeath> for StaticBirth {
    fn channel_count(
        &self,
        model: &TraitMutationBirthDeath,
        _state: &<TraitMutationBirthDeath as StaticFamilyModel>::State,
    ) -> usize {
        model.max_species
    }

    fn propensity(
        &self,
        _model: &TraitMutationBirthDeath,
        state: &<TraitMutationBirthDeath as StaticFamilyModel>::State,
        local_channel: usize,
        _time: f64,
    ) -> f64 {
        state
            .get(local_channel)
            .map_or(0.0, |sp| sp.birth_rate * sp.count as f64)
    }

    fn fire<R: Rng + ?Sized>(
        &self,
        _model: &TraitMutationBirthDeath,
        state: &mut <TraitMutationBirthDeath as StaticFamilyModel>::State,
        local_channel: usize,
        _rng: &mut R,
    ) -> TraitMutationEvent {
        if let Some(sp) = state.get_mut(local_channel) {
            sp.count += 1;
        }
        TraitMutationEvent {
            parent: local_channel,
            child: None,
        }
    }
}

impl StaticReactionFamily<TraitMutationBirthDeath> for StaticDeath {
    fn channel_count(
        &self,
        model: &TraitMutationBirthDeath,
        _state: &<TraitMutationBirthDeath as StaticFamilyModel>::State,
    ) -> usize {
        model.max_species
    }

    fn propensity(
        &self,
        _model: &TraitMutationBirthDeath,
        state: &<TraitMutationBirthDeath as StaticFamilyModel>::State,
        local_channel: usize,
        _time: f64,
    ) -> f64 {
        state
            .get(local_channel)
            .map_or(0.0, |sp| sp.death_rate * sp.count as f64)
    }

    fn fire<R: Rng + ?Sized>(
        &self,
        _model: &TraitMutationBirthDeath,
        state: &mut <TraitMutationBirthDeath as StaticFamilyModel>::State,
        local_channel: usize,
        _rng: &mut R,
    ) -> TraitMutationEvent {
        if let Some(sp) = state.get_mut(local_channel) {
            sp.count = sp.count.saturating_sub(1);
        }
        TraitMutationEvent {
            parent: local_channel,
            child: None,
        }
    }
}

impl StaticReactionFamily<TraitMutationBirthDeath> for StaticMutation {
    fn channel_count(
        &self,
        model: &TraitMutationBirthDeath,
        _state: &<TraitMutationBirthDeath as StaticFamilyModel>::State,
    ) -> usize {
        model.max_species
    }

    fn propensity(
        &self,
        _model: &TraitMutationBirthDeath,
        state: &<TraitMutationBirthDeath as StaticFamilyModel>::State,
        local_channel: usize,
        _time: f64,
    ) -> f64 {
        state
            .get(local_channel)
            .map_or(0.0, |sp| sp.mutation_rate * sp.count as f64)
    }

    fn fire<R: Rng + ?Sized>(
        &self,
        model: &TraitMutationBirthDeath,
        state: &mut <TraitMutationBirthDeath as StaticFamilyModel>::State,
        local_channel: usize,
        rng: &mut R,
    ) -> TraitMutationEvent {
        let Some(parent) = state.get(local_channel).copied() else {
            return TraitMutationEvent {
                parent: local_channel,
                child: None,
            };
        };
        state[local_channel].count = state[local_channel].count.saturating_sub(1);

        let child = state.len();
        let child = (child < model.max_species).then_some(child);
        if child.is_some() {
            state.push(TraitSpecies {
                count: 1,
                birth_rate: mutate_rate(parent.birth_rate, rng),
                death_rate: mutate_rate(parent.death_rate, rng),
                mutation_rate: mutate_rate(parent.mutation_rate, rng),
            });
        }
        TraitMutationEvent {
            parent: local_channel,
            child,
        }
    }
}

fn family_direct_trait_mutation_birth_death(
    max_species: usize,
    events: usize,
) -> (usize, u64, f64) {
    let model = trait_model(max_species);
    let mut simulation = FamilyDirect::new(model, family_list![
        StaticBirth,
        StaticDeath,
        StaticMutation
    ])
    .expect("model has valid propensities");
    let mut rngs = simulation_rngs(42);
    simulation
        .run(&mut rngs, events)
        .expect("model has valid updates");
    (
        simulation.state().len(),
        simulation.state().iter().map(|sp| sp.count).sum(),
        simulation.time(),
    )
}

fn family_nrm_trait_mutation_birth_death(max_species: usize, events: usize) -> (usize, u64, f64) {
    let model = trait_model(max_species);
    let mut rngs = simulation_rngs(42);
    let mut simulation = FamilyNrm::new(
        model,
        family_list![StaticBirth, StaticDeath, StaticMutation],
        rngs.clock_mut(),
    )
    .expect("model has valid propensities");
    simulation
        .run(&mut rngs, events)
        .expect("model has valid updates");
    (
        simulation.state().len(),
        simulation.state().iter().map(|sp| sp.count).sum(),
        simulation.time(),
    )
}

fn static_family_nrm_trait_mutation_birth_death(
    max_species: usize,
    events: usize,
) -> (usize, u64, f64) {
    let model = trait_model(max_species);
    let mut rngs = simulation_rngs(42);
    let mut simulation = StaticFamilyNrm::<_, _, 16>::new(
        model,
        family_list![StaticBirth, StaticDeath, StaticMutation],
        rngs.clock_mut(),
    )
    .expect("model has valid propensities");
    simulation
        .run(&mut rngs, events)
        .expect("model has valid updates");
    (
        simulation.state().len(),
        simulation.state().iter().map(|sp| sp.count).sum(),
        simulation.time(),
    )
}

fn bench_one_species(c: &mut Criterion) {
    let mut group = c.benchmark_group("one_species_birth_death");
    group.bench_function("handwritten", |b| {
        b.iter(|| handwritten_one_species_birth_death(black_box(ONE_SPECIES_EVENTS)))
    });
    group.bench_function("dynamic_direct", |b| {
        b.iter(|| dynamic_direct_one_species_birth_death(black_box(ONE_SPECIES_EVENTS)))
    });
    group.bench_function("family_direct", |b| {
        b.iter(|| family_direct_one_species_birth_death(black_box(ONE_SPECIES_EVENTS)))
    });
    group.bench_function("family_nrm", |b| {
        b.iter(|| family_nrm_one_species_birth_death(black_box(ONE_SPECIES_EVENTS)))
    });
    group.bench_function("static_family_nrm", |b| {
        b.iter(|| static_family_nrm_one_species_birth_death(black_box(ONE_SPECIES_EVENTS)))
    });
    group.finish();
}

fn bench_many_species(c: &mut Criterion) {
    let mut group = c.benchmark_group("many_species_birth_death");
    group.bench_function("handwritten", |b| {
        b.iter(|| {
            handwritten_many_species_birth_death(
                black_box(MANY_SPECIES),
                black_box(MANY_SPECIES_EVENTS),
            )
        })
    });
    group.bench_function("dynamic_direct", |b| {
        b.iter(|| {
            dynamic_direct_many_species_birth_death(
                black_box(MANY_SPECIES),
                black_box(MANY_SPECIES_EVENTS),
            )
        })
    });
    group.bench_function("family_direct", |b| {
        b.iter(|| {
            family_direct_many_species_birth_death(
                black_box(MANY_SPECIES),
                black_box(MANY_SPECIES_EVENTS),
            )
        })
    });
    group.bench_function("family_nrm", |b| {
        b.iter(|| {
            family_nrm_many_species_birth_death(
                black_box(MANY_SPECIES),
                black_box(MANY_SPECIES_EVENTS),
            )
        })
    });
    group.bench_function("static_family_nrm", |b| {
        b.iter(|| {
            static_family_nrm_many_species_birth_death(
                black_box(MANY_SPECIES),
                black_box(MANY_SPECIES_EVENTS),
            )
        })
    });
    group.finish();
}

fn bench_mutation_new_species(c: &mut Criterion) {
    let mut group = c.benchmark_group("mutation_new_species");
    group.bench_function("handwritten", |b| {
        b.iter(|| {
            handwritten_mutation_new_species(
                black_box(MUTATION_MAX_SPECIES),
                black_box(MUTATION_EVENTS),
            )
        })
    });
    group.bench_function("dynamic_direct", |b| {
        b.iter(|| {
            dynamic_direct_mutation_new_species(
                black_box(MUTATION_MAX_SPECIES),
                black_box(MUTATION_EVENTS),
            )
        })
    });
    group.bench_function("family_direct", |b| {
        b.iter(|| {
            family_direct_mutation_new_species(
                black_box(MUTATION_MAX_SPECIES),
                black_box(MUTATION_EVENTS),
            )
        })
    });
    group.bench_function("family_nrm", |b| {
        b.iter(|| {
            family_nrm_mutation_new_species(
                black_box(MUTATION_MAX_SPECIES),
                black_box(MUTATION_EVENTS),
            )
        })
    });
    group.bench_function("static_family_nrm", |b| {
        b.iter(|| {
            static_family_nrm_mutation_new_species(
                black_box(MUTATION_MAX_SPECIES),
                black_box(MUTATION_EVENTS),
            )
        })
    });
    group.finish();
}

fn bench_trait_mutation_birth_death(c: &mut Criterion) {
    let mut group = c.benchmark_group("trait_mutation_birth_death");
    group.bench_function("handwritten_family_dm", |b| {
        b.iter(|| {
            handwritten_trait_mutation_birth_death(
                black_box(MUTATION_MAX_SPECIES),
                black_box(TRAIT_MUTATION_EVENTS),
            )
        })
    });
    group.bench_function("handwritten_family_nrm", |b| {
        b.iter(|| {
            handwritten_family_nrm_trait_mutation_birth_death(
                black_box(MUTATION_MAX_SPECIES),
                black_box(TRAIT_MUTATION_EVENTS),
            )
        })
    });
    group.bench_function("family_direct", |b| {
        b.iter(|| {
            family_direct_trait_mutation_birth_death(
                black_box(MUTATION_MAX_SPECIES),
                black_box(TRAIT_MUTATION_EVENTS),
            )
        })
    });
    group.bench_function("family_nrm", |b| {
        b.iter(|| {
            family_nrm_trait_mutation_birth_death(
                black_box(MUTATION_MAX_SPECIES),
                black_box(TRAIT_MUTATION_EVENTS),
            )
        })
    });
    group.bench_function("static_family_nrm", |b| {
        b.iter(|| {
            static_family_nrm_trait_mutation_birth_death(
                black_box(MUTATION_MAX_SPECIES),
                black_box(TRAIT_MUTATION_EVENTS),
            )
        })
    });
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_one_species, bench_many_species, bench_mutation_new_species, bench_trait_mutation_birth_death
}
criterion_main!(benches);
