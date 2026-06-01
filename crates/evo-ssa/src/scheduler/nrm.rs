use rand::{Rng, RngExt};

use crate::{
    Error, Result,
    scheduler::{
        direct::{FamilySeparatedReactionSet, StaticDirectModel, StaticReactionFamily},
        update::{FamilyChannelUpdate, FamilyChannelUpdates, FixedFamilyChannelUpdates},
    },
};

/// Family-separated Next Reaction Method scheduler for piecewise-constant propensities.
///
/// Each concrete channel keeps its current propensity and scheduled firing time. After an event,
/// only dependency-reported channels are rescheduled. Unaffected channels keep their clocks.
#[derive(Debug, Clone)]
pub struct FamilyNrm<M, F>
where
    M: StaticDirectModel,
{
    model: M,
    state: M::State,
    families: F,
    family_offsets: Vec<usize>,
    clocks: Vec<FamilyClocks>,
    updates: Vec<FamilyChannelUpdate>,
    time: f64,
}

impl<M, F> FamilyNrm<M, F>
where
    M: StaticDirectModel,
    F: FamilySeparatedReactionSet<M>,
{
    pub fn new<R: Rng + ?Sized>(model: M, families: F, rng: &mut R) -> Result<Self> {
        let state = model.initial_state();
        let time = 0.0;
        let mut family_offsets = Vec::with_capacity(families.family_count());
        let mut clocks = Vec::with_capacity(families.family_count());
        let mut offset = 0;

        for family in 0..families.family_count() {
            family_offsets.push(offset);
            let channel_count = families.channel_count(&model, &state, family);
            let mut family_clocks = FamilyClocks::default();
            for local_channel in 0..channel_count {
                let propensity = checked_propensity(families.propensity(
                    &model,
                    &state,
                    family,
                    local_channel,
                    time,
                ))?;
                family_clocks.initialize(local_channel, propensity, time, rng);
            }
            offset += channel_count;
            clocks.push(family_clocks);
        }

        Ok(Self {
            model,
            state,
            families,
            family_offsets,
            clocks,
            updates: Vec::new(),
            time,
        })
    }

    pub fn time(&self) -> f64 {
        self.time
    }

    pub fn state(&self) -> &M::State {
        &self.state
    }

    pub fn step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<bool> {
        let Some((family, local_channel, scheduled_time)) = self.next_scheduled_channel() else {
            return Ok(false);
        };

        self.time = scheduled_time;
        self.updates.clear();
        let selected = FamilyChannelUpdate {
            family,
            local_channel,
        };
        let mut updates = FamilyChannelUpdates::new(&self.family_offsets, &mut self.updates);
        self.families
            .fire(&self.model, &mut self.state, selected, rng, &mut updates);

        self.reschedule_channel(family, local_channel, true, rng)?;
        while let Some(update) = self.updates.pop() {
            if update.family == family && update.local_channel == local_channel {
                continue;
            }
            self.reschedule_channel(update.family, update.local_channel, false, rng)?;
        }

        Ok(true)
    }

    pub fn run<R: Rng + ?Sized>(&mut self, rng: &mut R, max_events: usize) -> Result<usize> {
        let mut fired_events = 0;
        while fired_events < max_events && self.step(rng)? {
            fired_events += 1;
        }
        Ok(fired_events)
    }

    fn next_scheduled_channel(&self) -> Option<(usize, usize, f64)> {
        self.clocks
            .iter()
            .enumerate()
            .filter_map(|(family, clocks)| {
                let (local_channel, scheduled_time) = clocks.peek()?;
                Some((family, local_channel, scheduled_time))
            })
            .min_by(|(_, _, a), (_, _, b)| a.total_cmp(b))
    }

    fn reschedule_channel<R: Rng + ?Sized>(
        &mut self,
        family: usize,
        local_channel: usize,
        fired: bool,
        rng: &mut R,
    ) -> Result<()> {
        if family >= self.clocks.len() {
            return Ok(());
        }

        let propensity = checked_propensity(self.families.propensity(
            &self.model,
            &self.state,
            family,
            local_channel,
            self.time,
        ))?;
        self.clocks[family].reschedule(local_channel, propensity, self.time, fired, rng);
        Ok(())
    }
}

/// Rescheduling context passed to macro-generated static-family NRM implementations.
#[doc(hidden)]
pub struct StaticNrmReschedule<'a, M, R>
where
    M: StaticDirectModel,
    R: Rng + ?Sized,
{
    model: &'a M,
    state: &'a M::State,
    time: f64,
    fired: bool,
    rng: &'a mut R,
}

/// Compile-time reaction-family set used by [`StaticFamilyNrm`].
#[doc(hidden)]
pub trait StaticNrmFamilySet<M: StaticDirectModel> {
    type Clocks;

    fn family_count(&self) -> usize;

    fn initialize_clocks<R: Rng + ?Sized>(
        &self,
        model: &M,
        state: &M::State,
        time: f64,
        rng: &mut R,
    ) -> Result<(Vec<usize>, Self::Clocks)>;

    fn next_scheduled_channel(clocks: &Self::Clocks) -> Option<(usize, usize, f64)>;

    fn fire<R: Rng + ?Sized, const UPDATE_CAP: usize>(
        &self,
        model: &M,
        state: &mut M::State,
        selected: FamilyChannelUpdate,
        family_offsets: &[usize],
        rng: &mut R,
        updates: &mut FixedFamilyChannelUpdates<UPDATE_CAP>,
    );

    fn reschedule_channel<R: Rng + ?Sized>(
        &self,
        clocks: &mut Self::Clocks,
        family: usize,
        local_channel: usize,
        ctx: &mut StaticNrmReschedule<'_, M, R>,
    ) -> Result<()>;
}

/// Statically dispatched family-separated Next Reaction Method scheduler.
///
/// `F` is a tuple of compile-time reaction families. Tuple support is generated by macro for
/// several small arities, so this type is not tied to a particular workload such as
/// birth/death/mutation. `UPDATE_CAP` is the fixed stack buffer capacity for dependency updates
/// emitted by one fired event.
#[derive(Debug, Clone)]
pub struct StaticFamilyNrm<M, F, const UPDATE_CAP: usize>
where
    M: StaticDirectModel,
    F: StaticNrmFamilySet<M>,
{
    model: M,
    state: M::State,
    families: F,
    family_offsets: Vec<usize>,
    clocks: F::Clocks,
    updates: FixedFamilyChannelUpdates<UPDATE_CAP>,
    time: f64,
}

impl<M, F, const UPDATE_CAP: usize> StaticFamilyNrm<M, F, UPDATE_CAP>
where
    M: StaticDirectModel,
    F: StaticNrmFamilySet<M>,
{
    pub fn new<R: Rng + ?Sized>(model: M, families: F, rng: &mut R) -> Result<Self> {
        let state = model.initial_state();
        let time = 0.0;
        let (family_offsets, clocks) = families.initialize_clocks(&model, &state, time, rng)?;

        Ok(Self {
            model,
            state,
            families,
            family_offsets,
            clocks,
            updates: FixedFamilyChannelUpdates::default(),
            time,
        })
    }

    pub fn time(&self) -> f64 {
        self.time
    }

    pub fn state(&self) -> &M::State {
        &self.state
    }

    pub fn step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<bool> {
        let Some((family, local_channel, scheduled_time)) = F::next_scheduled_channel(&self.clocks)
        else {
            return Ok(false);
        };

        self.time = scheduled_time;
        self.updates.clear();
        self.families.fire(
            &self.model,
            &mut self.state,
            FamilyChannelUpdate {
                family,
                local_channel,
            },
            &self.family_offsets,
            rng,
            &mut self.updates,
        );

        self.reschedule_channel(family, local_channel, true, rng)?;
        while let Some(update) = self.updates.pop() {
            if update.family == family && update.local_channel == local_channel {
                continue;
            }
            self.reschedule_channel(update.family, update.local_channel, false, rng)?;
        }

        Ok(true)
    }

    pub fn run<R: Rng + ?Sized>(&mut self, rng: &mut R, max_events: usize) -> Result<usize> {
        let mut fired_events = 0;
        while fired_events < max_events && self.step(rng)? {
            fired_events += 1;
        }
        Ok(fired_events)
    }

    fn reschedule_channel<R: Rng + ?Sized>(
        &mut self,
        family: usize,
        local_channel: usize,
        fired: bool,
        rng: &mut R,
    ) -> Result<()> {
        let mut ctx = StaticNrmReschedule {
            model: &self.model,
            state: &self.state,
            time: self.time,
            fired,
            rng,
        };
        self.families
            .reschedule_channel(&mut self.clocks, family, local_channel, &mut ctx)
    }
}

macro_rules! impl_static_nrm_family_set {
    (@clock $ty:ident) => {
        FamilyClocks
    };

    ($count:expr; $( $idx:tt => $ty:ident : $clock:ident ),+ $(,)?) => {
        impl<M, $( $ty ),+> StaticNrmFamilySet<M> for ($( $ty, )+)
        where
            M: StaticDirectModel,
            $( $ty: StaticReactionFamily<M>, )+
        {
            type Clocks = ($(impl_static_nrm_family_set!(@clock $ty),)+);

            fn family_count(&self) -> usize {
                $count
            }

            fn initialize_clocks<R: Rng + ?Sized>(
                &self,
                model: &M,
                state: &M::State,
                time: f64,
                rng: &mut R,
            ) -> Result<(Vec<usize>, Self::Clocks)> {
                let mut family_offsets = Vec::with_capacity($count);
                let mut offset = 0;
                $(
                    family_offsets.push(offset);
                    offset += self.$idx.channel_count(model, state);
                    let mut $clock = FamilyClocks::default();
                    initialize_family_clocks(model, state, &self.$idx, &mut $clock, time, rng)?;
                )+
                let _ = offset;
                Ok((family_offsets, ($($clock,)+)))
            }

            fn next_scheduled_channel(clocks: &Self::Clocks) -> Option<(usize, usize, f64)> {
                let mut selected = None;
                $(
                    if let Some((local_channel, scheduled_time)) = clocks.$idx.peek()
                        && selected.is_none_or(|(_, _, selected_time)| scheduled_time < selected_time)
                    {
                        selected = Some(($idx, local_channel, scheduled_time));
                    }
                )+
                selected
            }

            fn fire<R: Rng + ?Sized, const UPDATE_CAP: usize>(
                &self,
                model: &M,
                state: &mut M::State,
                selected: FamilyChannelUpdate,
                family_offsets: &[usize],
                rng: &mut R,
                updates: &mut FixedFamilyChannelUpdates<UPDATE_CAP>,
            ) {
                match selected.family {
                    $(
                        $idx => self.$idx.fire(
                            model,
                            state,
                            selected.local_channel,
                            rng,
                            &mut updates.writer($idx, family_offsets[$idx], family_offsets),
                        ),
                    )+
                    _ => {}
                }
            }

            fn reschedule_channel<R: Rng + ?Sized>(
                &self,
                clocks: &mut Self::Clocks,
                family: usize,
                local_channel: usize,
                ctx: &mut StaticNrmReschedule<'_, M, R>,
            ) -> Result<()> {
                match family {
                    $(
                        $idx => {
                            let propensity = checked_propensity(self.$idx.propensity(
                                ctx.model,
                                ctx.state,
                                local_channel,
                                ctx.time,
                            ))?;
                            clocks.$idx.reschedule(
                                local_channel,
                                propensity,
                                ctx.time,
                                ctx.fired,
                                ctx.rng,
                            );
                        }
                    )+
                    _ => {}
                }
                Ok(())
            }
        }
    };
}

impl_static_nrm_family_set!(1; 0 => A: a);
impl_static_nrm_family_set!(2; 0 => A: a, 1 => B: b);
impl_static_nrm_family_set!(3; 0 => A: a, 1 => B: b, 2 => C: c);
impl_static_nrm_family_set!(4; 0 => A: a, 1 => B: b, 2 => C: c, 3 => D: d);
impl_static_nrm_family_set!(5; 0 => A: a, 1 => B: b, 2 => C: c, 3 => D: d, 4 => E: e);
impl_static_nrm_family_set!(6; 0 => A: a, 1 => B: b, 2 => C: c, 3 => D: d, 4 => E: e, 5 => G: g);

fn initialize_family_clocks<M, F, R>(
    model: &M,
    state: &M::State,
    family: &F,
    clocks: &mut FamilyClocks,
    time: f64,
    rng: &mut R,
) -> Result<()>
where
    M: StaticDirectModel,
    F: StaticReactionFamily<M>,
    R: Rng + ?Sized,
{
    for local_channel in 0..family.channel_count(model, state) {
        let propensity = checked_propensity(family.propensity(model, state, local_channel, time))?;
        clocks.initialize(local_channel, propensity, time, rng);
    }
    Ok(())
}

#[doc(hidden)]
#[derive(Debug, Clone, Default)]
pub struct FamilyClocks {
    clocks: Vec<Clock>,
    heap: IndexedMinHeap,
}

impl FamilyClocks {
    fn initialize<R: Rng + ?Sized>(
        &mut self,
        local_channel: usize,
        propensity: f64,
        now: f64,
        rng: &mut R,
    ) {
        self.ensure_clock(local_channel);
        let scheduled_time = draw_scheduled_time(now, propensity, rng);
        self.clocks[local_channel] = Clock {
            propensity,
            scheduled_time,
        };
        self.update_heap(local_channel, scheduled_time);
    }

    fn reschedule<R: Rng + ?Sized>(
        &mut self,
        local_channel: usize,
        new_propensity: f64,
        now: f64,
        fired: bool,
        rng: &mut R,
    ) {
        self.ensure_clock(local_channel);
        let old = self.clocks[local_channel];

        let scheduled_time = if new_propensity <= 0.0 {
            f64::INFINITY
        } else if fired || old.propensity <= 0.0 || !old.scheduled_time.is_finite() {
            draw_scheduled_time(now, new_propensity, rng)
        } else {
            let residual = (old.scheduled_time - now).max(0.0);
            now + (old.propensity / new_propensity) * residual
        };

        self.clocks[local_channel] = Clock {
            propensity: new_propensity,
            scheduled_time,
        };
        self.update_heap(local_channel, scheduled_time);
    }

    fn peek(&self) -> Option<(usize, f64)> {
        self.heap.peek().map(|local| {
            let scheduled_time = self.clocks[local].scheduled_time;
            (local, scheduled_time)
        })
    }

    fn ensure_clock(&mut self, local_channel: usize) {
        if local_channel >= self.clocks.len() {
            self.clocks.resize(local_channel + 1, Clock::inactive());
        }
    }

    fn update_heap(&mut self, local_channel: usize, scheduled_time: f64) {
        if scheduled_time.is_finite() {
            self.heap.insert_or_update(local_channel, scheduled_time);
        } else {
            self.heap.remove(local_channel);
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Clock {
    propensity: f64,
    scheduled_time: f64,
}

impl Clock {
    fn inactive() -> Self {
        Self {
            propensity: 0.0,
            scheduled_time: f64::INFINITY,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct IndexedMinHeap {
    heap: Vec<usize>,
    positions: Vec<Option<usize>>,
    times: Vec<f64>,
}

impl IndexedMinHeap {
    fn peek(&self) -> Option<usize> {
        self.heap.first().copied()
    }

    fn insert_or_update(&mut self, item: usize, time: f64) {
        self.ensure_item(item);
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
        if item >= self.positions.len() {
            return;
        }
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

    fn ensure_item(&mut self, item: usize) {
        if item >= self.positions.len() {
            self.positions.resize(item + 1, None);
            self.times.resize(item + 1, f64::INFINITY);
        }
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

fn draw_scheduled_time<R: Rng + ?Sized>(now: f64, propensity: f64, rng: &mut R) -> f64 {
    if propensity <= 0.0 {
        f64::INFINITY
    } else {
        now - rng.random::<f64>().clamp(f64::MIN_POSITIVE, 1.0).ln() / propensity
    }
}

fn checked_propensity(value: f64) -> Result<f64> {
    if value.is_finite() && value >= 0.0 {
        Ok(value)
    } else {
        Err(Error::InvalidPropensity { value })
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use rand::{SeedableRng, rngs::SmallRng};

    use super::*;
    use crate::scheduler::{
        direct::{StaticDirectModel, StaticReactionFamily},
        update::StaticUpdateSink,
    };

    #[test]
    fn indexed_heap_updates_minimum() {
        let mut heap = IndexedMinHeap::default();
        heap.insert_or_update(0, 3.0);
        heap.insert_or_update(1, 1.0);
        heap.insert_or_update(2, 2.0);
        assert_eq!(heap.peek(), Some(1));

        heap.insert_or_update(0, 0.5);
        assert_eq!(heap.peek(), Some(0));

        heap.remove(0);
        assert_eq!(heap.peek(), Some(1));
    }

    #[derive(Debug, Clone)]
    struct BirthDeathModel;

    impl StaticDirectModel for BirthDeathModel {
        type State = u64;

        fn initial_state(&self) -> Self::State {
            16
        }
    }

    #[derive(Debug, Clone, Copy)]
    struct Birth;

    #[derive(Debug, Clone, Copy)]
    struct Death;

    #[derive(Debug, Clone, Copy)]
    struct Inert;

    impl StaticReactionFamily<BirthDeathModel> for Birth {
        fn channel_count(&self, _model: &BirthDeathModel, _state: &u64) -> usize {
            1
        }

        fn propensity(
            &self,
            _model: &BirthDeathModel,
            state: &u64,
            _local_channel: usize,
            _time: f64,
        ) -> f64 {
            *state as f64
        }

        fn fire<R, U>(
            &self,
            _model: &BirthDeathModel,
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

    impl StaticReactionFamily<BirthDeathModel> for Death {
        fn channel_count(&self, _model: &BirthDeathModel, _state: &u64) -> usize {
            1
        }

        fn propensity(
            &self,
            _model: &BirthDeathModel,
            state: &u64,
            _local_channel: usize,
            _time: f64,
        ) -> f64 {
            0.25 * *state as f64
        }

        fn fire<R, U>(
            &self,
            _model: &BirthDeathModel,
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

    impl StaticReactionFamily<BirthDeathModel> for Inert {
        fn channel_count(&self, _model: &BirthDeathModel, _state: &u64) -> usize {
            1
        }

        fn propensity(
            &self,
            _model: &BirthDeathModel,
            _state: &u64,
            _local_channel: usize,
            _time: f64,
        ) -> f64 {
            0.0
        }

        fn fire<R, U>(
            &self,
            _model: &BirthDeathModel,
            _state: &mut u64,
            _local_channel: usize,
            _rng: &mut R,
            _updates: &mut U,
        ) where
            R: Rng + ?Sized,
            U: StaticUpdateSink + ?Sized,
        {
        }
    }

    #[test]
    fn static_family_nrm_runs_with_typed_update_sink() {
        let mut rng = SmallRng::seed_from_u64(7);
        let mut scheduler =
            StaticFamilyNrm::<_, _, 8>::new(BirthDeathModel, (Birth, Death, Inert), &mut rng)
                .expect("initial propensities are valid");

        let fired = scheduler
            .run(&mut rng, 64)
            .expect("updated propensities stay valid");

        assert_eq!(fired, 64);
        assert!(scheduler.time().is_finite());
        assert!(*scheduler.state() > 0);
    }
}
