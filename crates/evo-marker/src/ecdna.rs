use rand::{Rng, rngs::Xoshiro256PlusPlus};
use rand_distr::{Binomial, Distribution};

use crate::Marker;

/// A marker that stores the ecDNA copy number of a cell.
///
/// During cell division, a cell with `N` ecDNA copies first replicates to `2N` copies, then the
/// copies segregate to two daughters. Each copy independently chooses one daughter with
/// probability `0.5`, so one daughter receives `Binomial(2N, 0.5)` copies and the other receives
/// the remainder.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "bitcode", derive(bitcode::Encode, bitcode::Decode))]
pub struct EcDna {
    copy_number: u32,
}

impl EcDna {
    /// Create a cell with the given ecDNA copy number.
    pub const fn new(copy_number: u32) -> Self {
        Self { copy_number }
    }

    /// Return the ecDNA copy number carried by this cell.
    pub const fn copy_number(self) -> u32 {
        self.copy_number
    }
}

/// RNG type used by [`EcDna`] to sample copy-number segregation.
///
/// The marker itself does not derive or own RNG state. Reproducibility should be owned by the
/// caller, for example by passing in the stream created by `ssa-pipeline::StochasticStep`.
pub type EcDnaState = Xoshiro256PlusPlus;

impl Marker for EcDna {
    type State = EcDnaState;

    fn divide(&mut self, state: &mut Self::State) -> Self {
        debug_assert!(self.copy_number <= u32::MAX / 2);
        let total_copies = self.copy_number * 2;
        let daughter1 = sample_binomial_half(total_copies, state);
        let daughter2 = total_copies - daughter1;
        *self = Self::new(daughter1);
        Self::new(daughter2)
    }

    fn gen_marker(&self, _: &mut Self::State) -> Self {
        unreachable!("ecDNA daughters are jointly sampled; call divide instead of gen_marker")
    }
}

fn sample_binomial_half(n: u32, rng: &mut impl Rng) -> u32 {
    let distribution =
        Binomial::new(n as u64, 0.5).expect("Binomial(2N, 0.5) should always be valid");
    distribution.sample(rng) as u32
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use rand::SeedableRng;

    use super::*;

    #[test]
    fn division_replicates_then_partitions_total_copies() {
        let mut state = EcDnaState::seed_from_u64(0);
        let mut cell = EcDna::new(7);

        let daughter = cell.divide(&mut state);

        assert_eq!(cell.copy_number() + daughter.copy_number(), 14);
    }

    #[test]
    fn divide_at_increases_population_total_by_mother_copy_number() {
        let mut state = EcDnaState::seed_from_u64(0);
        let mut cells = vec![EcDna::new(32), EcDna::new(7), EcDna::new(3)];
        let mother_index = 1;
        let mother_copy_number = cells[mother_index].copy_number();
        let total_before: u32 = cells.iter().map(|cell| cell.copy_number()).sum();

        crate::divide_at(&mut cells, mother_index, &mut state);

        let total_after: u32 = cells.iter().map(|cell| cell.copy_number()).sum();
        assert_eq!(total_after, total_before + mother_copy_number);
    }

    #[test]
    fn division_is_reproducible_for_fixed_seed() {
        let mut state1 = EcDnaState::seed_from_u64(0);
        let mut state2 = EcDnaState::seed_from_u64(0);
        let mut cells1 = vec![EcDna::new(12), EcDna::new(8), EcDna::new(5)];
        let mut cells2 = cells1.clone();

        for &index in &[0, 1, 1, 3, 0] {
            crate::divide_at(&mut cells1, index, &mut state1);
            crate::divide_at(&mut cells2, index, &mut state2);
        }

        assert_eq!(cells1, cells2);
    }

    #[test]
    fn division_matches_binomial_mean() {
        let mut state = EcDnaState::seed_from_u64(0);
        let samples = 4_096;
        let total_daughter_copy_number: u64 = (0..samples)
            .map(|_| {
                let mut cell = EcDna::new(100);
                u64::from(cell.divide(&mut state).copy_number())
            })
            .sum();

        let mean = total_daughter_copy_number as f64 / samples as f64;
        assert!((mean - 100.0).abs() < 1.5);
    }

    #[cfg(feature = "bitcode")]
    #[test]
    fn encode_decode_roundtrip() {
        let marker = EcDna::new(123);
        let encoded = bitcode::encode(&marker);
        let decoded: EcDna = bitcode::decode(&encoded).unwrap();
        assert_eq!(decoded, marker);
    }

    #[test]
    #[should_panic(expected = "call divide instead of gen_marker")]
    fn gen_marker_is_not_supported() {
        let cell = EcDna::new(4);
        let mut state = EcDnaState::seed_from_u64(0);
        let _ = cell.gen_marker(&mut state);
    }
}
