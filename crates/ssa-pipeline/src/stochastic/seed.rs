use rand::{SeedableRng, rngs::Xoshiro256PlusPlus};

/// Seed domain used by [`super::StochasticStep`] constructors to derive their root seed.
pub const STOCHASTIC_ROOT_SEED_DOMAIN: SeedDomain =
    SeedDomain::new("ssa-pipeline/stochastic/root-seed/v1");

/// Stable identifier for deriving a root seed from caller-provided seed material.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SeedDomain(&'static str);

impl SeedDomain {
    /// Create a seed domain from a stable static name.
    pub const fn new(name: &'static str) -> Self {
        Self(name)
    }

    /// Return the domain name.
    pub const fn as_str(self) -> &'static str {
        self.0
    }
}

impl std::fmt::Display for SeedDomain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

/// Stable identifier for a reproducible random stream.
///
/// Domains are part of a stochastic protocol. Prefer crate- or subsystem-qualified names with a
/// version suffix, such as `ssa/main/v1` or `model/segregation/v1`.
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

/// Opaque root seed derived from a seed domain and caller-provided seed material.
///
/// A root seed can directly create the single legacy RNG stream. Domain-separated streams first
/// derive [`DomainSeed`] values from [`StreamDomain`] values.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct RootSeed {
    bytes: [u8; 32],
}

impl std::fmt::Debug for RootSeed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RootSeed").finish_non_exhaustive()
    }
}

impl RootSeed {
    /// Create a root seed from a stable seed domain and caller-provided seed material.
    pub fn from_domain(seed_domain: SeedDomain, seed_material: impl AsRef<[u8]>) -> Self {
        Self {
            bytes: blake3::derive_key(seed_domain.as_str(), seed_material.as_ref()),
        }
    }

    /// Derive the opaque seed for one stream domain.
    pub fn derive_domain_seed(&self, domain: StreamDomain) -> DomainSeed {
        DomainSeed {
            bytes: *blake3::keyed_hash(&self.bytes, domain.as_str().as_bytes()).as_bytes(),
        }
    }

    /// Derive an owned fixed-size bundle of domain seeds.
    pub fn derive_domain_seeds<const N: usize>(
        &self,
        domains: [StreamDomain; N],
    ) -> DomainSeeds<N> {
        DomainSeeds {
            seeds: domains.map(|domain| self.derive_domain_seed(domain)),
        }
    }

    /// Create the single legacy RNG stream for encoded input.
    ///
    /// This path intentionally does not use a [`StreamDomain`].
    pub fn make_stream(&self, encoded_input: &[u8]) -> Xoshiro256PlusPlus {
        let bytes = *blake3::keyed_hash(&self.bytes, encoded_input).as_bytes();
        Xoshiro256PlusPlus::from_seed(bytes)
    }
}

/// Opaque seed for one stream domain.
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
