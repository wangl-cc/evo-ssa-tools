use rand::Rng;

use crate::{Result, scheduler::math::draw_exponential};

/// Clock storage for one NRM reaction family.
#[doc(hidden)]
#[derive(Debug, Clone, Default)]
pub struct FamilyClocks {
    clocks: Vec<Clock>,
    heap: IndexedMinHeap,
}

impl FamilyClocks {
    pub(crate) fn len(&self) -> usize {
        self.clocks.len()
    }

    /// Resize while drawing clocks only for newly appended channels.
    pub(crate) fn resize_with<R: Rng + ?Sized>(
        &mut self,
        new_len: usize,
        now: f64,
        rng: &mut R,
        mut propensity: impl FnMut(usize) -> Result<f64>,
    ) -> Result<()> {
        let old_len = self.len();
        if new_len < old_len {
            for local_channel in new_len..old_len {
                self.heap.remove(local_channel);
            }
            self.clocks.truncate(new_len);
            return Ok(());
        }

        self.clocks.resize(new_len, Clock::inactive());
        for local_channel in old_len..new_len {
            self.initialize(local_channel, propensity(local_channel)?, now, rng);
        }
        Ok(())
    }

    pub(crate) fn reschedule<R: Rng + ?Sized>(
        &mut self,
        local_channel: usize,
        new_propensity: f64,
        now: f64,
        fired: bool,
        rng: &mut R,
    ) {
        let old = self.clocks[local_channel];
        let scheduled_time = if new_propensity <= 0.0 {
            f64::INFINITY
        } else if fired || old.propensity <= 0.0 || !old.scheduled_time.is_finite() {
            now + draw_exponential(new_propensity, rng)
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

    pub(crate) fn peek(&self) -> Option<(usize, f64)> {
        self.heap.peek().map(|local_channel| {
            let scheduled_time = self.clocks[local_channel].scheduled_time;
            (local_channel, scheduled_time)
        })
    }

    fn initialize<R: Rng + ?Sized>(
        &mut self,
        local_channel: usize,
        propensity: f64,
        now: f64,
        rng: &mut R,
    ) {
        let scheduled_time = now + draw_exponential(propensity, rng);
        self.clocks[local_channel] = Clock {
            propensity,
            scheduled_time,
        };
        self.update_heap(local_channel, scheduled_time);
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

        if let Some(position) = self.positions[item] {
            match time.total_cmp(&old_time) {
                std::cmp::Ordering::Less => self.sift_up(position),
                std::cmp::Ordering::Greater => self.sift_down(position),
                std::cmp::Ordering::Equal => {}
            }
        } else {
            let position = self.heap.len();
            self.heap.push(item);
            self.positions[item] = Some(position);
            self.sift_up(position);
        }
    }

    fn remove(&mut self, item: usize) {
        if item >= self.positions.len() {
            return;
        }
        let Some(position) = self.positions[item] else {
            return;
        };

        let last = self
            .heap
            .pop()
            .expect("stored position points into the heap");
        self.positions[item] = None;
        if position == self.heap.len() {
            return;
        }

        self.heap[position] = last;
        self.positions[last] = Some(position);
        self.sift_up(position);
        let position = self.positions[last].expect("sifted heap item retains a position");
        self.sift_down(position);
    }

    fn ensure_item(&mut self, item: usize) {
        if item >= self.positions.len() {
            self.positions.resize(item + 1, None);
            self.times.resize(item + 1, f64::INFINITY);
        }
    }

    fn sift_up(&mut self, mut position: usize) {
        while position > 0 {
            let parent = (position - 1) / 2;
            if self.less_or_equal(parent, position) {
                break;
            }
            self.swap_positions(parent, position);
            position = parent;
        }
    }

    fn sift_down(&mut self, mut position: usize) {
        loop {
            let left = position * 2 + 1;
            let right = left + 1;
            let mut smallest = position;

            if left < self.heap.len() && !self.less_or_equal(smallest, left) {
                smallest = left;
            }
            if right < self.heap.len() && !self.less_or_equal(smallest, right) {
                smallest = right;
            }
            if smallest == position {
                break;
            }

            self.swap_positions(position, smallest);
            position = smallest;
        }
    }

    fn less_or_equal(&self, a_position: usize, b_position: usize) -> bool {
        let a = self.heap[a_position];
        let b = self.heap[b_position];
        self.times[a].total_cmp(&self.times[b]).is_le()
    }

    fn swap_positions(&mut self, a: usize, b: usize) {
        self.heap.swap(a, b);
        self.positions[self.heap[a]] = Some(a);
        self.positions[self.heap[b]] = Some(b);
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use rand::{SeedableRng, rngs::SmallRng};

    use super::*;

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

    #[test]
    fn family_clocks_add_and_remove_channels_incrementally() -> Result<()> {
        let mut rng = SmallRng::seed_from_u64(11);
        let mut clocks = FamilyClocks::default();
        clocks.resize_with(3, 0.0, &mut rng, |_| Ok(1.0))?;
        assert_eq!(clocks.len(), 3);
        assert!(clocks.peek().is_some());

        clocks.resize_with(1, 0.0, &mut rng, |_| {
            unreachable!("shrinking does not evaluate propensities")
        })?;
        assert_eq!(clocks.len(), 1);
        assert!(clocks.peek().is_some_and(|(channel, _)| channel == 0));
        Ok(())
    }

    #[test]
    fn non_fired_clock_preserves_hazard_with_residual_update() {
        let mut clocks = FamilyClocks {
            clocks: vec![Clock {
                propensity: 2.0,
                scheduled_time: 5.0,
            }],
            heap: IndexedMinHeap::default(),
        };
        clocks.heap.insert_or_update(0, 5.0);
        let mut rng = SmallRng::seed_from_u64(13);

        clocks.reschedule(0, 4.0, 1.0, false, &mut rng);

        assert_eq!(clocks.clocks[0].scheduled_time, 3.0);
        assert_eq!(clocks.peek(), Some((0, 3.0)));
    }
}
