//! RNG stream identifiers and stream specifications.

use rand::{SeedableRng, rngs::Xoshiro256PlusPlus};

use crate::identity::{
    ComputationPath, IdentifierSegmentChain, append_len_prefixed, assert_identifier_segment,
};

const STREAM_SEED_CONTEXT: &str = "wangl-cc/evo-ssa-tools ssa-workflow stochastic stream seed v1";

/// Stable identifier for a model random variable with its own reproducible RNG stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RandomVariable(&'static str);

impl RandomVariable {
    /// Create a random variable identifier from a stable static name.
    ///
    /// Names use the same identifier segment rules as
    /// [`ComputationId`](crate::identity::ComputationId), except the empty name is reserved for the
    /// default unnamed stream.
    pub const fn new(name: &'static str) -> Self {
        assert_identifier_segment(name, true);
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

/// Single RNG stream specification.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SingleStream;

/// Named RNG stream specification.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NamedStreams<const N: usize> {
    variables: [RandomVariable; N],
}

impl<const N: usize> NamedStreams<N> {
    /// Create a named stream specification from stable random variable names.
    pub const fn new(variables: [RandomVariable; N]) -> Self {
        Self { variables }
    }

    /// Return the configured variables in stream order.
    pub const fn variables(&self) -> &[RandomVariable; N] {
        &self.variables
    }
}

impl ComputationPath {
    /// Derive the opaque seed for the default unnamed stream.
    pub fn derive_single_stream_seed(&self) -> StreamSeed {
        self.derive_stream_seed(RandomVariable::new(""))
    }

    /// Derive the opaque seed for one random variable stream.
    pub fn derive_stream_seed(&self, variable: RandomVariable) -> StreamSeed {
        derive_stream_seed(self, variable)
    }

    /// Derive an owned fixed-size bundle of stream seeds.
    pub fn derive_stream_seeds<const N: usize>(
        &self,
        variables: [RandomVariable; N],
    ) -> StreamSeeds<N> {
        StreamSeeds {
            seeds: variables.map(|variable| self.derive_stream_seed(variable)),
        }
    }
}

fn derive_stream_seed(path: &ComputationPath, variable: RandomVariable) -> StreamSeed {
    let mut material = Vec::with_capacity(128 + variable.as_str().len());
    append_len_prefixed(&mut material, &path.encode_segments());
    append_len_prefixed(&mut material, variable.as_str().as_bytes());
    StreamSeed::derive_from(&material)
}

#[doc(hidden)]
pub trait SeedSource {
    type Rng;

    fn make_rng(&self, encoded_input: &[u8]) -> Self::Rng;
}

#[doc(hidden)]
pub trait StreamSpecSeed {
    type Seed: SeedSource;

    fn derive_seed(&self, path: &ComputationPath) -> Self::Seed;
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
    pub(crate) fn derive_from(material: &[u8]) -> Self {
        Self {
            bytes: blake3::derive_key(STREAM_SEED_CONTEXT, material),
        }
    }

    /// Create a fresh RNG stream for this seed and encoded input.
    pub fn make_stream(&self, encoded_input: &[u8]) -> Xoshiro256PlusPlus {
        let bytes = blake3::keyed_hash(&self.bytes, encoded_input).into();
        Xoshiro256PlusPlus::from_seed(bytes)
    }
}

impl SeedSource for StreamSeed {
    type Rng = Xoshiro256PlusPlus;

    fn make_rng(&self, encoded_input: &[u8]) -> Self::Rng {
        self.make_stream(encoded_input)
    }
}

/// Owned fixed-size bundle of stream seeds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamSeeds<const N: usize> {
    pub(crate) seeds: [StreamSeed; N],
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

impl<const N: usize> SeedSource for StreamSeeds<N> {
    type Rng = RngStreams<N>;

    fn make_rng(&self, encoded_input: &[u8]) -> Self::Rng {
        self.make_streams(encoded_input)
    }
}

impl StreamSpecSeed for SingleStream {
    type Seed = StreamSeed;

    fn derive_seed(&self, path: &ComputationPath) -> Self::Seed {
        path.derive_single_stream_seed()
    }
}

impl<const N: usize> StreamSpecSeed for NamedStreams<N> {
    type Seed = StreamSeeds<N>;

    fn derive_seed(&self, path: &ComputationPath) -> Self::Seed {
        path.derive_stream_seeds(*self.variables())
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
    use crate::identity::ComputationId;

    const MAIN_VARIABLE: RandomVariable = RandomVariable::new("ssa-main-v1");
    const SEGREGATION_VARIABLE: RandomVariable = RandomVariable::new("model-segregation-v1");
    const MUTATION_VARIABLE: RandomVariable = RandomVariable::new("model-mutation-v1");

    fn test_path() -> ComputationPath {
        ComputationPath::root_from_str("experiment-ssa-workflow-test-v1")
    }

    mod identifiers {
        use super::*;

        #[test]
        fn names_are_stable() {
            let model = ComputationId::new("experiment-test-v1");
            let variable = RandomVariable::new("test-stream-v1");

            assert_eq!(model.as_str(), "experiment-test-v1");
            assert_eq!(variable.as_str(), "test-stream-v1");
            assert_eq!(model.to_string(), "experiment-test-v1");
            assert_eq!(variable.to_string(), "test-stream-v1");
        }
    }

    mod stream_seed {
        use super::*;

        #[test]
        fn debug_output_is_redacted() {
            let stream_seed = test_path().derive_stream_seed(MAIN_VARIABLE);

            let stream_debug = format!("{stream_seed:?}");

            assert_eq!(stream_debug, "StreamSeed { .. }");
            assert!(!stream_debug.contains("bytes"));
        }

        #[test]
        fn streams_are_stable_and_isolated() {
            let path = test_path();
            let segregation_seed = path.derive_stream_seed(SEGREGATION_VARIABLE);
            let mutation_seed = path.derive_stream_seed(MUTATION_VARIABLE);
            let mut segregation1 = segregation_seed.make_stream(b"input-A");
            let mut segregation2 = segregation_seed.make_stream(b"input-A");
            let mut mutation = mutation_seed.make_stream(b"input-A");

            assert_eq!(segregation1.next_u64(), segregation2.next_u64());
            assert_ne!(segregation1.next_u64(), mutation.next_u64());
        }

        #[test]
        fn input_bytes_change_stream() {
            let seed = test_path().derive_stream_seed(SEGREGATION_VARIABLE);
            let mut rng_a = seed.make_stream(b"input-A");
            let mut rng_b = seed.make_stream(b"input-B");

            assert_ne!(rng_a.next_u64(), rng_b.next_u64());
        }

        #[test]
        fn material_boundaries_change_stream() {
            let seed_ab_c =
                ComputationPath::root_from_str("ab").derive_stream_seed(RandomVariable::new("c"));
            let seed_a_bc =
                ComputationPath::root_from_str("a").derive_stream_seed(RandomVariable::new("bc"));
            let mut rng_ab_c = seed_ab_c.make_stream(b"input-A");
            let mut rng_a_bc = seed_a_bc.make_stream(b"input-A");

            assert_ne!(rng_ab_c.next_u64(), rng_a_bc.next_u64());
        }

        #[test]
        fn single_stream_known_sequence() {
            let seed = test_path().derive_single_stream_seed();
            let mut rng = seed.make_stream(b"input-A");

            assert_eq!(rng.next_u64(), 1_421_434_473_334_949_077);
            assert_eq!(rng.next_u64(), 3_224_070_076_799_729_687);
        }

        #[test]
        fn named_stream_known_sequence() {
            let seed = test_path().derive_stream_seed(SEGREGATION_VARIABLE);
            let mut rng = seed.make_stream(b"input-A");

            let first = rng.next_u64();
            let second = rng.next_u64();
            assert_eq!(first, 4_391_430_058_245_016_709);
            assert_eq!(second, 1_974_120_400_265_464_438);
        }

        #[test]
        fn single_stream_is_isolated_from_named_stream() {
            let path = test_path();
            let single_seed = path.derive_single_stream_seed();
            let stream_seed = path.derive_stream_seed(MAIN_VARIABLE);
            let mut single_rng = single_seed.make_stream(b"input-A");
            let mut stream_rng = stream_seed.make_stream(b"input-A");

            assert_ne!(single_rng.next_u64(), stream_rng.next_u64());
        }

        #[test]
        fn single_stream_is_the_empty_random_variable() {
            let path = test_path();
            let single_seed = path.derive_single_stream_seed();
            let empty_variable_seed = path.derive_stream_seed(RandomVariable::new(""));
            let mut single_rng = single_seed.make_stream(b"input-A");
            let mut empty_variable_rng = empty_variable_seed.make_stream(b"input-A");

            assert_eq!(single_rng.next_u64(), empty_variable_rng.next_u64());
        }
    }

    mod bundles {
        use super::*;

        #[test]
        fn stream_seed_bundle_reuses_the_seed_for_duplicate_variable_names() {
            let seeds =
                test_path().derive_stream_seeds([SEGREGATION_VARIABLE, SEGREGATION_VARIABLE]);
            let mut streams = seeds.make_streams(b"input-A");
            let [left, right] = streams.as_mut();

            assert_eq!(left.next_u64(), right.next_u64());
        }

        #[test]
        fn bundled_stream_seeds_match_individually_derived_variable_seeds() {
            let path = test_path();
            let direct_segregation = path.derive_stream_seed(SEGREGATION_VARIABLE);
            let direct_mutation = path.derive_stream_seed(MUTATION_VARIABLE);

            let seeds = path.derive_stream_seeds([SEGREGATION_VARIABLE, MUTATION_VARIABLE]);
            let [bundled_segregation, bundled_mutation] = seeds.as_ref();

            assert_eq!(*bundled_segregation, direct_segregation);
            assert_eq!(*bundled_mutation, direct_mutation);
        }

        #[test]
        fn adding_or_reordering_named_streams_does_not_change_each_variable_seed() {
            let path = test_path();
            let direct_segregation = path.derive_stream_seed(SEGREGATION_VARIABLE);
            let direct_mutation = path.derive_stream_seed(MUTATION_VARIABLE);

            let expanded =
                path.derive_stream_seeds([MAIN_VARIABLE, SEGREGATION_VARIABLE, MUTATION_VARIABLE]);
            let [_main, expanded_segregation, expanded_mutation] = expanded.into_inner();
            assert_eq!(expanded_segregation, direct_segregation);
            assert_eq!(expanded_mutation, direct_mutation);

            let reordered = path.derive_stream_seeds([MUTATION_VARIABLE, SEGREGATION_VARIABLE]);
            let [reordered_mutation, reordered_segregation] = reordered.into_inner();
            assert_eq!(reordered_segregation, direct_segregation);
            assert_eq!(reordered_mutation, direct_mutation);
        }

        #[test]
        fn stream_seed_bundle_accessors_preserve_order() {
            let seeds = test_path().derive_stream_seeds([SEGREGATION_VARIABLE, MUTATION_VARIABLE]);
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
            let seeds = test_path().derive_stream_seeds([SEGREGATION_VARIABLE, MUTATION_VARIABLE]);
            let mut streams = seeds.make_streams(b"input-A");
            let [segregation_rng, mutation_rng] = streams.as_mut();

            let segregation_value = segregation_rng.next_u64();
            let mutation_value = mutation_rng.next_u64();

            assert_ne!(segregation_value, mutation_value);
        }

        #[test]
        fn rng_stream_bundle_into_inner_preserves_order() {
            let seeds = test_path().derive_stream_seeds([SEGREGATION_VARIABLE, MUTATION_VARIABLE]);
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
