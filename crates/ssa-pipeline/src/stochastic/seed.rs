use rand::{SeedableRng, rngs::Xoshiro256PlusPlus};

const SINGLE_STREAM_DOMAIN: StreamDomain =
    StreamDomain::new("ssa-pipeline/stochastic/single-stream/v1");

/// Stable namespace for one stochastic experiment or model protocol.
///
/// The experiment domain is part of reproducibility, not a per-run random seed. Use a stable,
/// versioned name such as `experiment/cell-growth/v1`. Different random trajectories within the
/// same experiment should use different [`super::StochasticInput`] repetition indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExperimentDomain(&'static str);

impl ExperimentDomain {
    /// Create an experiment domain from a stable static name.
    pub const fn new(name: &'static str) -> Self {
        Self(name)
    }

    /// Return the domain name.
    pub const fn as_str(self) -> &'static str {
        self.0
    }

    /// Derive the opaque seed for the single-stream stochastic protocol.
    pub fn derive_single_stream_seed(self) -> DomainSeed {
        self.derive_domain_seed(SINGLE_STREAM_DOMAIN)
    }

    /// Derive the opaque seed for one random stream domain.
    pub fn derive_domain_seed(self, domain: StreamDomain) -> DomainSeed {
        DomainSeed {
            bytes: blake3::derive_key(domain.as_str(), self.as_str().as_bytes()),
        }
    }

    /// Derive an owned fixed-size bundle of domain seeds.
    pub fn derive_domain_seeds<const N: usize>(self, domains: [StreamDomain; N]) -> DomainSeeds<N> {
        DomainSeeds {
            seeds: domains.map(|domain| self.derive_domain_seed(domain)),
        }
    }
}

impl std::fmt::Display for ExperimentDomain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

/// Stable identifier for a reproducible random stream.
///
/// Domains are part of a stochastic protocol. Prefer crate- or subsystem-qualified names with a
/// version suffix, such as `cell-model/division-event/v1` or
/// `cell-model/copy-number-segregation/v1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StreamDomain(&'static str);

impl StreamDomain {
    /// Create a stream domain from a stable static name.
    pub const fn new(name: &'static str) -> Self {
        Self(name)
    }

    /// Return the domain name.
    pub const fn as_str(self) -> &'static str {
        self.0
    }
}

impl std::fmt::Display for StreamDomain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

/// Opaque seed for one stream domain in one experiment domain.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct DomainSeed {
    bytes: [u8; 32],
}

impl std::fmt::Debug for DomainSeed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DomainSeed").finish_non_exhaustive()
    }
}

impl DomainSeed {
    /// Create a fresh RNG for this domain seed and encoded input.
    pub fn make_stream(&self, encoded_input: &[u8]) -> Xoshiro256PlusPlus {
        let bytes = *blake3::keyed_hash(&self.bytes, encoded_input).as_bytes();
        Xoshiro256PlusPlus::from_seed(bytes)
    }
}

/// Owned fixed-size bundle of domain seeds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DomainSeeds<const N: usize> {
    seeds: [DomainSeed; N],
}

impl<const N: usize> DomainSeeds<N> {
    /// Consume the bundle and return the owned domain-seed array.
    pub fn into_inner(self) -> [DomainSeed; N] {
        self.seeds
    }

    /// Create an owned fixed-size bundle of RNG streams for encoded input.
    pub fn make_streams(&self, encoded_input: &[u8]) -> StochasticStreams<N> {
        StochasticStreams {
            rngs: self.seeds.map(|seed| seed.make_stream(encoded_input)),
        }
    }
}

impl<const N: usize> AsRef<[DomainSeed; N]> for DomainSeeds<N> {
    fn as_ref(&self) -> &[DomainSeed; N] {
        &self.seeds
    }
}

/// Owned fixed-size RNG bundle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StochasticStreams<const N: usize> {
    rngs: [Xoshiro256PlusPlus; N],
}

impl<const N: usize> StochasticStreams<N> {
    /// Consume the bundle and return the owned RNG array.
    pub fn into_inner(self) -> [Xoshiro256PlusPlus; N] {
        self.rngs
    }
}

impl<const N: usize> AsMut<[Xoshiro256PlusPlus; N]> for StochasticStreams<N> {
    fn as_mut(&mut self) -> &mut [Xoshiro256PlusPlus; N] {
        &mut self.rngs
    }
}
