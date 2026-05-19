use rand::{SeedableRng, rngs::Xoshiro256PlusPlus};

const STREAM_SEED_CONTEXT: &str = "wangl-cc/evo-ssa-tools ssa-pipeline stochastic stream seed v1";
const SINGLE_STREAM_VARIABLE: RandomVariable = RandomVariable::new("");

/// Stable identifier for one stochastic simulation model.
///
/// Use a stable, versioned name such as `birth-death-ssa/v1`. The simulation model is combined
/// with random variables and [`super::StochasticInput`] to derive reproducible RNG streams.
/// Stream seeds use a length-prefixed encoding of `(SimulationModel, RandomVariable)` as BLAKE3 key
/// material. The single-stream seed is derived with the empty random variable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SimulationModel(&'static str);

impl SimulationModel {
    /// Create a simulation model identifier from a stable static name.
    pub const fn new(name: &'static str) -> Self {
        Self(name)
    }

    /// Return the model name.
    pub const fn as_str(self) -> &'static str {
        self.0
    }

    /// Derive the opaque seed for the single-stream stochastic model.
    pub fn derive_single_stream_seed(self) -> StreamSeed {
        self.derive_stream_seed(SINGLE_STREAM_VARIABLE)
    }

    /// Derive the opaque seed for one model random variable.
    pub fn derive_stream_seed(self, variable: RandomVariable) -> StreamSeed {
        StreamSeed {
            bytes: blake3::derive_key(
                STREAM_SEED_CONTEXT,
                &encode_stream_seed_material(self, variable),
            ),
        }
    }

    /// Derive an owned fixed-size bundle of stream seeds.
    pub fn derive_stream_seeds<const N: usize>(
        self,
        variables: [RandomVariable; N],
    ) -> StreamSeeds<N> {
        StreamSeeds {
            seeds: variables.map(|variable| self.derive_stream_seed(variable)),
        }
    }
}

fn encode_stream_seed_material(model: SimulationModel, variable: RandomVariable) -> Vec<u8> {
    let model = model.as_str().as_bytes();
    let variable = variable.as_str().as_bytes();

    [&(model.len() as u64).to_be_bytes(), model, variable].concat()
}

impl std::fmt::Display for SimulationModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

/// Stable identifier for a model random variable with its own reproducible RNG stream.
///
/// Prefer model-specific names with a version suffix, such as `ssa/waiting-time/v1` or
/// `ssa/reaction-choice/v1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RandomVariable(&'static str);

impl RandomVariable {
    /// Create a random variable identifier from a stable static name.
    pub const fn new(name: &'static str) -> Self {
        Self(name)
    }

    /// Return the variable name.
    pub const fn as_str(self) -> &'static str {
        self.0
    }
}

impl std::fmt::Display for RandomVariable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

/// Opaque seed for one RNG stream in one simulation model.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct StreamSeed {
    bytes: [u8; 32],
}

impl std::fmt::Debug for StreamSeed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamSeed").finish_non_exhaustive()
    }
}

impl StreamSeed {
    /// Create a fresh RNG stream for this seed and encoded input.
    pub fn make_stream(&self, encoded_input: &[u8]) -> Xoshiro256PlusPlus {
        let bytes = blake3::keyed_hash(&self.bytes, encoded_input).into();
        Xoshiro256PlusPlus::from_seed(bytes)
    }
}

/// Owned fixed-size bundle of stream seeds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamSeeds<const N: usize> {
    seeds: [StreamSeed; N],
}

impl<const N: usize> StreamSeeds<N> {
    /// Consume the bundle and return the owned stream-seed array.
    pub fn into_inner(self) -> [StreamSeed; N] {
        self.seeds
    }

    /// Create an owned fixed-size bundle of RNG streams for encoded input.
    pub fn make_streams(&self, encoded_input: &[u8]) -> RngStreams<N> {
        RngStreams {
            rngs: self.seeds.map(|seed| seed.make_stream(encoded_input)),
        }
    }
}

impl<const N: usize> AsRef<[StreamSeed; N]> for StreamSeeds<N> {
    fn as_ref(&self) -> &[StreamSeed; N] {
        &self.seeds
    }
}

/// Owned fixed-size RNG stream bundle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RngStreams<const N: usize> {
    rngs: [Xoshiro256PlusPlus; N],
}

impl<const N: usize> RngStreams<N> {
    /// Consume the bundle and return the owned RNG array.
    pub fn into_inner(self) -> [Xoshiro256PlusPlus; N] {
        self.rngs
    }
}

impl<const N: usize> AsMut<[Xoshiro256PlusPlus; N]> for RngStreams<N> {
    fn as_mut(&mut self) -> &mut [Xoshiro256PlusPlus; N] {
        &mut self.rngs
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use rand::Rng;

    use super::*;

    const TEST_MODEL: SimulationModel = SimulationModel::new("experiment/ssa-pipeline-test/v1");
    const MAIN_VARIABLE: RandomVariable = RandomVariable::new("ssa/main/v1");
    const SEGREGATION_VARIABLE: RandomVariable = RandomVariable::new("model/segregation/v1");
    const MUTATION_VARIABLE: RandomVariable = RandomVariable::new("model/mutation/v1");

    mod identifiers {
        use super::*;

        #[test]
        fn names_are_stable() {
            let model = SimulationModel::new("experiment/test/v1");
            let variable = RandomVariable::new("test/stream/v1");

            assert_eq!(model.as_str(), "experiment/test/v1");
            assert_eq!(variable.as_str(), "test/stream/v1");
            assert_eq!(model.to_string(), "experiment/test/v1");
            assert_eq!(variable.to_string(), "test/stream/v1");
        }
    }

    mod stream_seed {
        use super::*;

        #[test]
        fn debug_output_is_redacted() {
            let stream_seed = TEST_MODEL.derive_stream_seed(MAIN_VARIABLE);

            let stream_debug = format!("{stream_seed:?}");

            assert_eq!(stream_debug, "StreamSeed { .. }");
            assert!(!stream_debug.contains("bytes"));
        }

        #[test]
        fn streams_are_stable_and_isolated() {
            let segregation_seed = TEST_MODEL.derive_stream_seed(SEGREGATION_VARIABLE);
            let mutation_seed = TEST_MODEL.derive_stream_seed(MUTATION_VARIABLE);
            let mut segregation1 = segregation_seed.make_stream(b"input-A");
            let mut segregation2 = segregation_seed.make_stream(b"input-A");
            let mut mutation = mutation_seed.make_stream(b"input-A");

            assert_eq!(segregation1.next_u64(), segregation2.next_u64());
            assert_ne!(segregation1.next_u64(), mutation.next_u64());
        }

        #[test]
        fn input_bytes_change_stream() {
            let seed = TEST_MODEL.derive_stream_seed(SEGREGATION_VARIABLE);
            let mut rng_a = seed.make_stream(b"input-A");
            let mut rng_b = seed.make_stream(b"input-B");

            assert_ne!(rng_a.next_u64(), rng_b.next_u64());
        }

        #[test]
        fn material_boundaries_change_stream() {
            let seed_ab_c = SimulationModel::new("ab").derive_stream_seed(RandomVariable::new("c"));
            let seed_a_bc = SimulationModel::new("a").derive_stream_seed(RandomVariable::new("bc"));
            let mut rng_ab_c = seed_ab_c.make_stream(b"input-A");
            let mut rng_a_bc = seed_a_bc.make_stream(b"input-A");

            assert_ne!(rng_ab_c.next_u64(), rng_a_bc.next_u64());
        }

        #[test]
        fn single_stream_known_sequence() {
            let seed = TEST_MODEL.derive_single_stream_seed();
            let mut rng = seed.make_stream(b"input-A");

            assert_eq!(rng.next_u64(), 18_439_914_891_317_024_306);
            assert_eq!(rng.next_u64(), 3_187_733_765_017_413_158);
        }

        #[test]
        fn named_stream_known_sequence() {
            let seed = TEST_MODEL.derive_stream_seed(SEGREGATION_VARIABLE);
            let mut rng = seed.make_stream(b"input-A");

            assert_eq!(rng.next_u64(), 14_685_298_993_141_689_080);
            assert_eq!(rng.next_u64(), 5_115_129_566_991_615_718);
        }

        #[test]
        fn single_stream_is_isolated_from_named_stream() {
            let single_seed = TEST_MODEL.derive_single_stream_seed();
            let stream_seed = TEST_MODEL.derive_stream_seed(MAIN_VARIABLE);
            let mut single_rng = single_seed.make_stream(b"input-A");
            let mut stream_rng = stream_seed.make_stream(b"input-A");

            assert_ne!(single_rng.next_u64(), stream_rng.next_u64());
        }

        #[test]
        fn single_stream_matches_empty_random_variable() {
            let single_seed = TEST_MODEL.derive_single_stream_seed();
            let empty_variable_seed = TEST_MODEL.derive_stream_seed(RandomVariable::new(""));
            let mut single_rng = single_seed.make_stream(b"input-A");
            let mut empty_variable_rng = empty_variable_seed.make_stream(b"input-A");

            assert_eq!(single_rng.next_u64(), empty_variable_rng.next_u64());
        }
    }

    mod bundles {
        use super::*;

        #[test]
        fn stream_seed_bundle_allows_duplicate_variables() {
            let seeds =
                TEST_MODEL.derive_stream_seeds([SEGREGATION_VARIABLE, SEGREGATION_VARIABLE]);
            let mut streams = seeds.make_streams(b"input-A");
            let [left, right] = streams.as_mut();

            assert_eq!(left.next_u64(), right.next_u64());
        }

        #[test]
        fn stream_seed_bundle_accessors_preserve_order() {
            let seeds = TEST_MODEL.derive_stream_seeds([SEGREGATION_VARIABLE, MUTATION_VARIABLE]);
            let [segregation_seed_ref, mutation_seed_ref] = seeds.as_ref();
            let mut segregation_rng = segregation_seed_ref.make_stream(b"input-A");
            let mut mutation_rng = mutation_seed_ref.make_stream(b"input-A");

            let [segregation_seed, mutation_seed] = seeds.into_inner();
            let mut owned_segregation_rng = segregation_seed.make_stream(b"input-A");
            let mut owned_mutation_rng = mutation_seed.make_stream(b"input-A");

            assert_eq!(segregation_rng.next_u64(), owned_segregation_rng.next_u64());
            assert_eq!(mutation_rng.next_u64(), owned_mutation_rng.next_u64());
        }

        #[test]
        fn stream_seed_bundle_supports_multiple_mutable_rngs() {
            let seeds = TEST_MODEL.derive_stream_seeds([SEGREGATION_VARIABLE, MUTATION_VARIABLE]);
            let mut streams = seeds.make_streams(b"input-A");
            let [segregation_rng, mutation_rng] = streams.as_mut();

            let segregation_value = segregation_rng.next_u64();
            let mutation_value = mutation_rng.next_u64();

            assert_ne!(segregation_value, mutation_value);
        }

        #[test]
        fn rng_stream_bundle_into_inner_preserves_order() {
            let seeds = TEST_MODEL.derive_stream_seeds([SEGREGATION_VARIABLE, MUTATION_VARIABLE]);
            let mut streams = seeds.make_streams(b"input-A");
            let [segregation_rng, mutation_rng] = streams.as_mut();

            let segregation_value = segregation_rng.next_u64();
            let mutation_value = mutation_rng.next_u64();

            let streams = seeds.make_streams(b"input-A");
            let [mut owned_segregation_rng, mut owned_mutation_rng] = streams.into_inner();

            assert_eq!(segregation_value, owned_segregation_rng.next_u64());
            assert_eq!(mutation_value, owned_mutation_rng.next_u64());
        }
    }
}
