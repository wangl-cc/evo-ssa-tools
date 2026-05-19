use rand::{SeedableRng, rngs::Xoshiro256PlusPlus};

const SINGLE_STREAM_VARIABLE: RandomVariable =
    RandomVariable::new("ssa-pipeline/stochastic/single-stream");

/// Stable identifier for one stochastic simulation model.
///
/// Use a stable, versioned name such as `birth-death-ssa/v1`. The simulation model is combined
/// with random variables and [`super::StochasticInput`] to derive reproducible RNG streams.
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
            bytes: blake3::derive_key(variable.as_str(), self.as_str().as_bytes()),
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
        let bytes = *blake3::keyed_hash(&self.bytes, encoded_input).as_bytes();
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
