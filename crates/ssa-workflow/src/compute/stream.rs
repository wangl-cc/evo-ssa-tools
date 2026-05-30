//! Random stream labels and RNG bundles for stochastic workflows.

use rand::{SeedableRng, rngs::Xoshiro256PlusPlus};

use crate::identity::{ComputationPath, IdentifierSegmentChain, SEGMENT_ENCODED_SEPARATOR};

const STREAM_SEED_CONTEXT: &str = "wangl-cc/evo-ssa-tools ssa-workflow stochastic stream seed v1";

/// Stable name for one random variable in a stochastic computation.
///
/// Use the same name whenever you want the same random variable to keep the
/// same reproducible stream. Changing a name changes that stream's random
/// sequence. Stream names are seed labels only; they are not part of persistent
/// storage namespaces.
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

/// Describes which random streams a stochastic function receives.
///
/// Choose this through builder methods such as
/// `StochasticTask::builder(...).streams(["waiting", "choice"])`.
pub trait StreamSpec: private::Sealed {
    /// Seed material prepared when the task or transform is built.
    type Seed: SeedSource;

    /// Prepare seed material for this stream spec and computation path.
    fn derive_seed(&self, path: &ComputationPath) -> Self::Seed;
}

/// Prepared seed material for stochastic tasks and transforms.
///
/// Most users do not need to name this trait. It connects a stream spec to the
/// RNG value passed to the stochastic function for one input.
pub trait SeedSource: private::Sealed + Clone + Send + Sync + 'static {
    /// Runtime RNG argument passed to the user function.
    type Rng: 'static;

    /// Create the RNG for one encoded input.
    fn make_rng(&self, encoded_input: &[u8]) -> Self::Rng;
}

/// Single-stream RNG specification.
///
/// This is the default. The stochastic function receives one
/// `Xoshiro256PlusPlus` RNG.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SingleStream;

impl StreamSpec for SingleStream {
    type Seed = StreamSeed;

    fn derive_seed(&self, path: &ComputationPath) -> Self::Seed {
        path.derive_seed(RandomVariable::new(""))
    }
}

impl ComputationPath {
    /// Prepare the seed for one named random variable stream.
    pub fn derive_seed(&self, variable: RandomVariable) -> StreamSeed {
        let mut hasher = blake3::Hasher::new_derive_key(STREAM_SEED_CONTEXT);
        self.hash_segments(&mut hasher);
        hasher.update(SEGMENT_ENCODED_SEPARATOR);
        hasher.update(variable.as_str().as_bytes());
        let bytes = hasher.finalize().into();
        StreamSeed { bytes }
    }
}

/// Multi-stream RNG specification.
///
/// Use this when a stochastic function needs several independent named random streams.
/// The function receives an [`RngBundle`] in the same order as the configured names.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MultiStreams<const N: usize> {
    variables: [RandomVariable; N],
}

impl<const N: usize> MultiStreams<N> {
    /// Create a multi-stream specification from stable random variable names.
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

/// Prepared seed for one random stream.
///
/// The same `StreamSeed` and encoded input always produce the same RNG sequence.
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

/// Prepared seeds for a multi-stream stochastic function.
///
/// The bundle stores one [`StreamSeed`] per configured random variable,
/// in the same order as the corresponding [`MultiStreams`] specification.
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

/// Fixed-size bundle of RNGs.
///
/// This is passed to multi-stream stochastic functions. Use `as_mut` to destructure the bundle
/// into mutable RNG references.
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
    use std::sync::LazyLock;

    use rand::Rng;

    use super::*;
    use crate::identity::ComputationId;

    const MAIN_VARIABLE: RandomVariable = RandomVariable::new("main");
    const SEGREGATION_VARIABLE: RandomVariable = RandomVariable::new("segregation");
    const MUTATION_VARIABLE: RandomVariable = RandomVariable::new("mutation");

    static TEST_PATH: LazyLock<ComputationPath> =
        LazyLock::new(|| ComputationPath::root_from_str("experiment-ssa-workflow-test-v1"));

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

        #[test]
        fn multi_stream_keep_order() {
            let streams = MultiStreams::from(["x", "y", "z"]);
            assert_eq!(streams.variables(), &[
                RandomVariable::new("x"),
                RandomVariable::new("y"),
                RandomVariable::new("z"),
            ]);
        }
    }

    mod stream_seed {
        use super::*;

        #[test]
        fn debug_output_is_redacted() {
            let single_stream_seed = SingleStream.derive_seed(&TEST_PATH);
            let stream_debug = format!("{single_stream_seed:?}");
            assert_eq!(stream_debug, "StreamSeed { .. }");

            let multi_stream = MultiStreams::from(["x", "y", "z"]);
            let multi_stream_seed = multi_stream.derive_seed(&TEST_PATH);
            let multi_stream_debug = format!("{multi_stream_seed:?}");
            assert_eq!(multi_stream_debug, "StreamSeed { .. }");
        }

        #[test]
        fn streams_are_stable_and_isolated() {
            let segregation_seed = TEST_PATH.derive_seed(SEGREGATION_VARIABLE);
            let mutation_seed = TEST_PATH.derive_seed(MUTATION_VARIABLE);
            let mut segregation1 = segregation_seed.make_rng(b"input-A");
            let mut segregation2 = segregation_seed.make_rng(b"input-A");
            let mut mutation = mutation_seed.make_rng(b"input-A");

            assert_eq!(segregation1.next_u64(), segregation2.next_u64());
            assert_ne!(segregation1.next_u64(), mutation.next_u64());
        }

        #[test]
        fn input_bytes_change_stream() {
            let seed = TEST_PATH.derive_seed(SEGREGATION_VARIABLE);
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
            let seed = SingleStream.derive_seed(&TEST_PATH);
            let mut rng = seed.make_rng(b"input-A");

            assert_eq!(rng.next_u64(), 9757323776284558303);
            assert_eq!(rng.next_u64(), 14831252709949991980);
        }

        #[test]
        fn named_stream_known_sequence() {
            let seed = TEST_PATH.derive_seed(SEGREGATION_VARIABLE);
            let mut rng = seed.make_rng(b"input-A");

            let first = rng.next_u64();
            let second = rng.next_u64();
            assert_eq!(first, 14830145518112621704);
            assert_eq!(second, 12584622603152527866);
        }

        #[test]
        fn single_stream_is_isolated_from_named_stream() {
            let stream_seed = TEST_PATH.derive_seed(MAIN_VARIABLE);
            let mut single_rng = SingleStream.derive_seed(&TEST_PATH).make_rng(b"input-A");
            let mut stream_rng = stream_seed.make_rng(b"input-A");

            assert_ne!(single_rng.next_u64(), stream_rng.next_u64());
        }

        #[test]
        fn single_stream_is_the_empty_random_variable() {
            let empty_variable_seed = TEST_PATH.derive_seed(RandomVariable::new(""));
            let mut single_rng = SingleStream.derive_seed(&TEST_PATH).make_rng(b"input-A");
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
                .derive_seed(&TEST_PATH)
                .make_rng(b"input-A");
            let [left, right] = streams.as_mut();

            assert_eq!(left.next_u64(), right.next_u64());
        }

        #[test]
        fn named_streams_match_individually_derived_variable_streams() {
            let mut direct_segregation = TEST_PATH
                .derive_seed(SEGREGATION_VARIABLE)
                .make_rng(b"input-A");
            let mut direct_mutation = TEST_PATH
                .derive_seed(MUTATION_VARIABLE)
                .make_rng(b"input-A");

            let mut streams = MultiStreams::new([SEGREGATION_VARIABLE, MUTATION_VARIABLE])
                .derive_seed(&TEST_PATH)
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
            let mut direct_segregation = TEST_PATH
                .derive_seed(SEGREGATION_VARIABLE)
                .make_rng(b"input-A");
            let mut direct_mutation = TEST_PATH
                .derive_seed(MUTATION_VARIABLE)
                .make_rng(b"input-A");

            let mut expanded =
                MultiStreams::new([MAIN_VARIABLE, SEGREGATION_VARIABLE, MUTATION_VARIABLE])
                    .derive_seed(&TEST_PATH)
                    .make_rng(b"input-A");
            let [_main, expanded_segregation, expanded_mutation] = expanded.as_mut();
            assert_eq!(
                expanded_segregation.next_u64(),
                direct_segregation.next_u64()
            );
            assert_eq!(expanded_mutation.next_u64(), direct_mutation.next_u64());

            let mut direct_segregation = TEST_PATH
                .derive_seed(SEGREGATION_VARIABLE)
                .make_rng(b"input-A");
            let mut direct_mutation = TEST_PATH
                .derive_seed(MUTATION_VARIABLE)
                .make_rng(b"input-A");
            let mut reordered = MultiStreams::new([MUTATION_VARIABLE, SEGREGATION_VARIABLE])
                .derive_seed(&TEST_PATH)
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
                .derive_seed(&TEST_PATH)
                .make_rng(b"input-A");
            let [segregation_rng, mutation_rng] = streams.as_mut();

            let segregation_value = segregation_rng.next_u64();
            let mutation_value = mutation_rng.next_u64();

            assert_ne!(segregation_value, mutation_value);
        }

        #[test]
        fn rng_stream_bundle_as_mut_preserves_order() {
            let mut streams = MultiStreams::new([SEGREGATION_VARIABLE, MUTATION_VARIABLE])
                .derive_seed(&TEST_PATH)
                .make_rng(b"input-A");
            let [segregation_rng, mutation_rng] = streams.as_mut();

            let segregation_value = segregation_rng.next_u64();
            let mutation_value = mutation_rng.next_u64();

            let mut streams = MultiStreams::new([SEGREGATION_VARIABLE, MUTATION_VARIABLE])
                .derive_seed(&TEST_PATH)
                .make_rng(b"input-A");
            let [owned_segregation_rng, owned_mutation_rng] = streams.as_mut();

            assert_eq!(segregation_value, owned_segregation_rng.next_u64());
            assert_eq!(mutation_value, owned_mutation_rng.next_u64());
        }
    }
}
