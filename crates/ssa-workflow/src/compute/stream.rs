//! RNG stream identifiers and stream specifications.

use rand::{SeedableRng, rngs::Xoshiro256PlusPlus};

use crate::identity::{ComputationPath, IdentifierSegmentChain, SEGMENT_ENCODED_SEPARATOR};

const STREAM_SEED_CONTEXT: &str = "wangl-cc/evo-ssa-tools ssa-workflow stochastic stream seed v1";

/// Stable label for a model random variable with its own reproducible RNG stream.
///
/// Random variable names are opaque seed labels, not storage namespace segments.
/// Changing a label changes the derived stream for that variable.
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

mod private {
    use super::{MultiStreams, SingleStream, StreamSeed, StreamSeeds};

    pub trait Sealed {}

    impl Sealed for SingleStream {}
    impl<const N: usize> Sealed for MultiStreams<N> {}
    impl Sealed for StreamSeed {}
    impl<const N: usize> Sealed for StreamSeeds<N> {}
}

/// RNG stream used by stochastic tasks and transforms.
pub trait StreamSpec: private::Sealed {
    /// Build-time seed value stored by the compute node.
    type Seed: SeedSource;

    /// Derive the build-time seed value for this stream spec and computation path.
    fn derive_seed(&self, path: &ComputationPath) -> Self::Seed;
}

/// Seed material for stochastic tasks and transforms.
///
/// A seed source is derived once while building a compute node. Each execution
/// then calls [`SeedSource::make_rng`] with the canonical encoded input bytes to
/// create the runtime RNG passed to the user function.
pub trait SeedSource: private::Sealed + Clone + Send + Sync + 'static {
    /// Runtime RNG argument passed to the user function.
    type Rng: 'static;

    /// Create the runtime RNG for one canonical encoded input.
    fn make_rng(&self, encoded_input: &[u8]) -> Self::Rng;
}

/// Single-stream RNG specification.
///
/// Used when you only need one RNG stream.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SingleStream;

impl StreamSpec for SingleStream {
    type Seed = StreamSeed;

    fn derive_seed(&self, path: &ComputationPath) -> Self::Seed {
        path.derive_seed(RandomVariable::new(""))
    }
}

impl ComputationPath {
    /// Derive the opaque seed for one random variable stream.
    pub fn derive_seed(&self, variable: RandomVariable) -> StreamSeed {
        let mut hasher = blake3::Hasher::new_derive_key(STREAM_SEED_CONTEXT);
        self.hash_segments(&mut hasher);
        hasher.update(SEGMENT_ENCODED_SEPARATOR);
        hasher.update(variable.as_str().as_bytes());
        let bytes = hasher.finalize().into();
        StreamSeed { bytes }
    }
}

/// Multi-stream RNG specification
///
/// Used when you need multiple RNG streams. Each stream have its own name.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MultiStreams<const N: usize> {
    variables: [RandomVariable; N],
}

impl<const N: usize> MultiStreams<N> {
    /// Create a named stream specification from stable random variable names.
    pub const fn new(variables: [RandomVariable; N]) -> Self {
        Self { variables }
    }

    /// Return the configured variables in stream order.
    pub const fn variables(&self) -> &[RandomVariable; N] {
        &self.variables
    }
}

impl<const N: usize> From<[RandomVariable; N]> for MultiStreams<N> {
    fn from(variables: [RandomVariable; N]) -> Self {
        Self::new(variables)
    }
}

impl<const N: usize> From<[&'static str; N]> for MultiStreams<N> {
    fn from(names: [&'static str; N]) -> Self {
        Self::new(names.map(RandomVariable::new))
    }
}

impl<const N: usize> StreamSpec for MultiStreams<N> {
    type Seed = StreamSeeds<N>;

    fn derive_seed(&self, path: &ComputationPath) -> Self::Seed {
        let seeds = self.variables.map(|variable| path.derive_seed(variable));
        StreamSeeds { seeds }
    }
}

/// Opaque seed for one RNG stream in one simulation model.
///
/// A `StreamSeed` is derived from a computation path and random variable name.
/// It is not itself an RNG, combines it with the canonical encoded input bytes for one execution.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct StreamSeed {
    bytes: [u8; 32],
}

impl std::fmt::Debug for StreamSeed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamSeed").finish_non_exhaustive()
    }
}

impl SeedSource for StreamSeed {
    type Rng = Xoshiro256PlusPlus;

    fn make_rng(&self, encoded_input: &[u8]) -> Self::Rng {
        let bytes = blake3::keyed_hash(&self.bytes, encoded_input).into();
        Xoshiro256PlusPlus::from_seed(bytes)
    }
}

/// Opaque fixed-size seed bundle for named RNG streams.
///
/// The bundle stores one [`StreamSeed`] per configured random variable,
/// in the same order as the corresponding [`MultiStream`] specification.
/// Like `StreamSeed`, this is build-time seed material; each execution mixes
/// in the canonical encoded input bytes before constructing runtime RNGs.
#[derive(Clone, PartialEq, Eq)]
pub struct StreamSeeds<const N: usize> {
    seeds: [StreamSeed; N],
}

impl<const N: usize> std::fmt::Debug for StreamSeeds<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamSeeds").finish_non_exhaustive()
    }
}

impl<const N: usize> SeedSource for StreamSeeds<N> {
    type Rng = RngBundle<N>;

    fn make_rng(&self, encoded_input: &[u8]) -> Self::Rng {
        let rngs = self.seeds.map(|seed| seed.make_rng(encoded_input));
        RngBundle { rngs }
    }
}

/// Owned fixed-size RNG bundle.
///
/// It's contains multiple [`Xoshiro256PlusPlus`] RNGs, one per configured random variable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RngBundle<const N: usize> {
    rngs: [Xoshiro256PlusPlus; N],
}

impl<const N: usize> AsMut<[Xoshiro256PlusPlus; N]> for RngBundle<N> {
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

    const MAIN_VARIABLE: RandomVariable = RandomVariable::new("main");
    const SEGREGATION_VARIABLE: RandomVariable = RandomVariable::new("segregation");
    const MUTATION_VARIABLE: RandomVariable = RandomVariable::new("mutation");

    fn test_path() -> ComputationPath {
        ComputationPath::root_from_str("experiment-ssa-workflow-test-v1")
    }

    mod identifiers {
        use super::*;

        #[test]
        fn names_are_stable() {
            let model = ComputationId::new("experiment-test-v1");
            let variable = RandomVariable::new("αβγ/ℝⁿ/𝔼[X_t]/λ₁→∞");

            assert_eq!(model.as_str(), "experiment-test-v1");
            assert_eq!(variable.as_str(), "αβγ/ℝⁿ/𝔼[X_t]/λ₁→∞");
            assert_eq!(model.to_string(), "experiment-test-v1");
            assert_eq!(variable.to_string(), "αβγ/ℝⁿ/𝔼[X_t]/λ₁→∞");
        }
    }

    mod stream_seed {
        use super::*;

        #[test]
        fn debug_output_is_redacted() {
            let stream_seed = test_path().derive_seed(MAIN_VARIABLE);

            let stream_debug = format!("{stream_seed:?}");

            assert_eq!(stream_debug, "StreamSeed { .. }");
            assert!(!stream_debug.contains("bytes"));
        }

        #[test]
        fn streams_are_stable_and_isolated() {
            let path = test_path();
            let segregation_seed = path.derive_seed(SEGREGATION_VARIABLE);
            let mutation_seed = path.derive_seed(MUTATION_VARIABLE);
            let mut segregation1 = segregation_seed.make_rng(b"input-A");
            let mut segregation2 = segregation_seed.make_rng(b"input-A");
            let mut mutation = mutation_seed.make_rng(b"input-A");

            assert_eq!(segregation1.next_u64(), segregation2.next_u64());
            assert_ne!(segregation1.next_u64(), mutation.next_u64());
        }

        #[test]
        fn input_bytes_change_stream() {
            let seed = test_path().derive_seed(SEGREGATION_VARIABLE);
            let mut rng_a = seed.make_rng(b"input-A");
            let mut rng_b = seed.make_rng(b"input-B");

            assert_ne!(rng_a.next_u64(), rng_b.next_u64());
        }

        #[test]
        fn material_boundaries_change_stream() {
            let seed_ab_c =
                ComputationPath::root_from_str("ab").derive_seed(RandomVariable::new("c"));
            let seed_a_bc =
                ComputationPath::root_from_str("a").derive_seed(RandomVariable::new("bc"));
            let mut rng_ab_c = seed_ab_c.make_rng(b"input-A");
            let mut rng_a_bc = seed_a_bc.make_rng(b"input-A");

            assert_ne!(rng_ab_c.next_u64(), rng_a_bc.next_u64());
        }

        #[test]
        fn single_stream_known_sequence() {
            let seed = SingleStream.derive_seed(&test_path());
            let mut rng = seed.make_rng(b"input-A");

            assert_eq!(rng.next_u64(), 9757323776284558303);
            assert_eq!(rng.next_u64(), 14831252709949991980);
        }

        #[test]
        fn named_stream_known_sequence() {
            let seed = test_path().derive_seed(SEGREGATION_VARIABLE);
            let mut rng = seed.make_rng(b"input-A");

            let first = rng.next_u64();
            let second = rng.next_u64();
            assert_eq!(first, 14830145518112621704);
            assert_eq!(second, 12584622603152527866);
        }

        #[test]
        fn single_stream_is_isolated_from_named_stream() {
            let path = test_path();
            let stream_seed = path.derive_seed(MAIN_VARIABLE);
            let mut single_rng = SingleStream.derive_seed(&path).make_rng(b"input-A");
            let mut stream_rng = stream_seed.make_rng(b"input-A");

            assert_ne!(single_rng.next_u64(), stream_rng.next_u64());
        }

        #[test]
        fn single_stream_is_the_empty_random_variable() {
            let path = test_path();
            let empty_variable_seed = path.derive_seed(RandomVariable::new(""));
            let mut single_rng = SingleStream.derive_seed(&path).make_rng(b"input-A");
            let mut empty_variable_rng = empty_variable_seed.make_rng(b"input-A");

            assert_eq!(single_rng.next_u64(), empty_variable_rng.next_u64());
        }
    }

    mod rng_bundle {
        use super::*;

        #[test]
        fn from_str_array_matches_new() {
            let from_names = MultiStreams::from(["segregation", "mutation"]);
            let from_new = MultiStreams::new([SEGREGATION_VARIABLE, MUTATION_VARIABLE]);

            assert_eq!(from_names, from_new);
        }

        #[test]
        fn from_variable_array_matches_new() {
            let from_variables = MultiStreams::from([SEGREGATION_VARIABLE, MUTATION_VARIABLE]);
            let from_new = MultiStreams::new([SEGREGATION_VARIABLE, MUTATION_VARIABLE]);

            assert_eq!(from_variables, from_new);
        }

        #[test]
        fn named_streams_reuse_the_seed_for_duplicate_variable_names() {
            let mut streams = MultiStreams::new([SEGREGATION_VARIABLE, SEGREGATION_VARIABLE])
                .derive_seed(&test_path())
                .make_rng(b"input-A");
            let [left, right] = streams.as_mut();

            assert_eq!(left.next_u64(), right.next_u64());
        }

        #[test]
        fn named_streams_match_individually_derived_variable_streams() {
            let path = test_path();
            let mut direct_segregation =
                path.derive_seed(SEGREGATION_VARIABLE).make_rng(b"input-A");
            let mut direct_mutation = path.derive_seed(MUTATION_VARIABLE).make_rng(b"input-A");

            let mut streams = MultiStreams::new([SEGREGATION_VARIABLE, MUTATION_VARIABLE])
                .derive_seed(&path)
                .make_rng(b"input-A");
            let [bundled_segregation, bundled_mutation] = streams.as_mut();

            assert_eq!(
                bundled_segregation.next_u64(),
                direct_segregation.next_u64()
            );
            assert_eq!(bundled_mutation.next_u64(), direct_mutation.next_u64());
        }

        #[test]
        fn adding_or_reordering_named_streams_does_not_change_each_variable_stream() {
            let path = test_path();
            let mut direct_segregation =
                path.derive_seed(SEGREGATION_VARIABLE).make_rng(b"input-A");
            let mut direct_mutation = path.derive_seed(MUTATION_VARIABLE).make_rng(b"input-A");

            let mut expanded =
                MultiStreams::new([MAIN_VARIABLE, SEGREGATION_VARIABLE, MUTATION_VARIABLE])
                    .derive_seed(&path)
                    .make_rng(b"input-A");
            let [_main, expanded_segregation, expanded_mutation] = expanded.as_mut();
            assert_eq!(
                expanded_segregation.next_u64(),
                direct_segregation.next_u64()
            );
            assert_eq!(expanded_mutation.next_u64(), direct_mutation.next_u64());

            let mut direct_segregation =
                path.derive_seed(SEGREGATION_VARIABLE).make_rng(b"input-A");
            let mut direct_mutation = path.derive_seed(MUTATION_VARIABLE).make_rng(b"input-A");
            let mut reordered = MultiStreams::new([MUTATION_VARIABLE, SEGREGATION_VARIABLE])
                .derive_seed(&path)
                .make_rng(b"input-A");
            let [reordered_mutation, reordered_segregation] = reordered.as_mut();
            assert_eq!(
                reordered_segregation.next_u64(),
                direct_segregation.next_u64()
            );
            assert_eq!(reordered_mutation.next_u64(), direct_mutation.next_u64());
        }

        #[test]
        fn rng_bundle_supports_multiple_mutable_rngs() {
            let mut streams = MultiStreams::new([SEGREGATION_VARIABLE, MUTATION_VARIABLE])
                .derive_seed(&test_path())
                .make_rng(b"input-A");
            let [segregation_rng, mutation_rng] = streams.as_mut();

            let segregation_value = segregation_rng.next_u64();
            let mutation_value = mutation_rng.next_u64();

            assert_ne!(segregation_value, mutation_value);
        }

        #[test]
        fn rng_stream_bundle_as_mut_preserves_order() {
            let path = test_path();
            let mut streams = MultiStreams::new([SEGREGATION_VARIABLE, MUTATION_VARIABLE])
                .derive_seed(&path)
                .make_rng(b"input-A");
            let [segregation_rng, mutation_rng] = streams.as_mut();

            let segregation_value = segregation_rng.next_u64();
            let mutation_value = mutation_rng.next_u64();

            let mut streams = MultiStreams::new([SEGREGATION_VARIABLE, MUTATION_VARIABLE])
                .derive_seed(&path)
                .make_rng(b"input-A");
            let [owned_segregation_rng, owned_mutation_rng] = streams.as_mut();

            assert_eq!(segregation_value, owned_segregation_rng.next_u64());
            assert_eq!(mutation_value, owned_mutation_rng.next_u64());
        }
    }
}
