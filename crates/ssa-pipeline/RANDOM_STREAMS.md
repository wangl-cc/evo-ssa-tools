# Random Streams Design

This document describes the design for reproducible named random streams in `ssa-pipeline`.

## Motivation

`StochasticStep::new` gives each input one deterministic `Xoshiro256PlusPlus` stream. That model is simple and correct for simulations where randomness is naturally consumed as one sequential trajectory.

Some simulations need stronger engineering stability. A model may contain several random subsystems such as event scheduling, mutation sampling, copy-number segregation, and final sampling. If all of them consume one shared stream, changing the implementation of one subsystem can shift the random values consumed by later subsystems. The result is still reproducible for the exact same code, but it is sensitive to code ordering and local refactors.

The long-term goal is to support module-level stream isolation without making random seed management the responsibility of each simulation component.

## Core Model

Seed derivation starts from a root seed:

```text
SeedDomain + seed_material
-> RootSeed

RootSeed + encoded_input_bytes
-> Xoshiro256PlusPlus
```

The single-stream path intentionally has no stream domain. Domain-separated streams add one more seed derivation stage:

```text
RootSeed + StreamDomain
-> DomainSeed

DomainSeed + encoded_input_bytes
-> Xoshiro256PlusPlus
```

These stages intentionally use different types in the implementation. A `RootSeed` is not a domain seed, and a `DomainSeed` is not an advancing RNG.

Conceptually, execution looks like:

```text
StochasticStep
-> RootSeed
-> StochasticInput
-> canonical encoded input
-> RootSeed::make_stream
-> Xoshiro256PlusPlus
```

For domain-separated execution, `StochasticStep::new_with_domain_streams` pre-derives `DomainSeeds<N>` from the configured domains and uses those domain seeds to create `StochasticStreams<N>` for each encoded input.

## Seed Types

Seed handling lives under `stochastic::seed`.

```rust
pub struct SeedDomain(&'static str);
pub struct StreamDomain(&'static str);
pub struct RootSeed {
    // private fields
}
pub struct DomainSeed {
    // private fields
}
pub struct DomainSeeds<const N: usize> {
    // private fields
}

impl RootSeed {
    pub fn from_domain(seed_domain: SeedDomain, seed_material: impl AsRef<[u8]>) -> Self;
    pub fn derive_domain_seed(&self, domain: StreamDomain) -> DomainSeed;
    pub fn derive_domain_seeds<const N: usize>(&self, domains: [StreamDomain; N]) -> DomainSeeds<N>;
    pub fn make_stream(&self, encoded_input: &[u8]) -> Xoshiro256PlusPlus;
}

impl DomainSeed {
    pub fn make_stream(&self, encoded_input: &[u8]) -> Xoshiro256PlusPlus;
}

impl<const N: usize> DomainSeeds<N> {
    pub fn make_streams(&self, encoded_input: &[u8]) -> StochasticStreams<N>;
}
```

`RootSeed`, `DomainSeed`, and `DomainSeeds<N>` are opaque and do not expose raw seed bytes. `derive_domain_seed` and `derive_domain_seeds` are named as derivation operations because they create domain-specific seeds rather than fetching a stored domain object. `RootSeed::make_stream` is the single-stream path and intentionally does not use a `StreamDomain`.

## StochasticStep API

The single-stream constructor remains:

```rust
let step = StochasticStep::new(
    store,
    seed_material,
    |rng, param| {
        // simulation logic using the default stream
        Ok(output)
    },
    engine_factory,
);
```

This uses `DEFAULT_SEED_DOMAIN` to derive the root seed, then derives the one RNG stream directly from the encoded input.

The domain-stream constructor is:

```rust
pub const DIVISION_EVENT_STREAM: StreamDomain = StreamDomain::new("cell-model/division-event/v1");
pub const COPY_NUMBER_SEGREGATION_STREAM: StreamDomain =
    StreamDomain::new("cell-model/copy-number-segregation/v1");

let step = StochasticStep::new_with_domain_streams(
    store,
    seed_material,
    [DIVISION_EVENT_STREAM, COPY_NUMBER_SEGREGATION_STREAM],
    |rngs, parent_copy_number| {
        let [division_rng, segregation_rng] = rngs.as_mut();

        let divides = sample_division_event(division_rng);
        if !divides {
            return Ok((parent_copy_number, 0));
        }

        let replicated = parent_copy_number * 2;
        let left_daughter = sample_copy_number_split(segregation_rng, replicated);
        let right_daughter = replicated - left_daughter;

        Ok((left_daughter, right_daughter))
    },
    engine_factory,
);
```

The difference is domain stream construction, not a different seed model. Both constructors use the same root seed derivation policy. The domain-stream constructor derives `DomainSeeds<N>` once at construction and creates input-scoped `StochasticStreams<N>` during execution.

Internally, both modes share the same `StochasticStep` execution shell, but the stored `seed` type differs:

```rust
StochasticStep<C, P, O, RootSeed, F, EF>
StochasticStep<C, P, O, DomainSeeds<N>, F, EF>
```

The field is intentionally named `seed`. Whether it is a single root seed or a fixed bundle of domain seeds is expressed by the generic type.

## Stream Domains

A stream domain is a stable protocol identifier:

```rust
pub struct StreamDomain(&'static str);
```

Domains should be constants, not ad hoc strings spread through simulation code:

```rust
pub const DIVISION_EVENT_STREAM: StreamDomain = StreamDomain::new("cell-model/division-event/v1");
pub const COPY_NUMBER_SEGREGATION_STREAM: StreamDomain =
    StreamDomain::new("cell-model/copy-number-segregation/v1");
pub const MUTATION_SAMPLING_STREAM: StreamDomain =
    StreamDomain::new("cell-model/mutation-sampling/v1");
```

Domain names should include the owning crate or system, the subsystem, and a version. Changing the random protocol for a subsystem should use a new domain version instead of silently changing the meaning of an existing domain.

## Seed Derivation

Substreams are derived using deterministic domain separation:

```text
root_seed = derive_key(seed_domain, seed_material)
single_stream_seed = keyed_hash(root_seed, encoded_input_bytes)
single_stream_rng = Xoshiro256PlusPlus::from_seed(single_stream_seed)

domain_seed = keyed_hash(root_seed, stream_domain_bytes)
domain_stream_seed = keyed_hash(domain_seed, encoded_input_bytes)
domain_stream_rng = Xoshiro256PlusPlus::from_seed(domain_stream_seed)
```

This is intentionally seed derivation, not RNG sampling. The domain `cell-model/copy-number-segregation/v1` maps to the same domain seed regardless of whether other domains are requested before or after it.

This also means `SeedableRng::fork` is not the right core primitive for domain streams. `fork` is useful for creating another RNG from a current RNG state, but it is order-sensitive. Adding a new fork before an existing fork changes the later child stream.

`jump` and `long_jump` are also not the right general abstraction for this crate. They are useful when a chosen RNG exposes them, but the `rand` crate's exported `Xoshiro256PlusPlus` does not expose a generic stream-splitting trait. Domain-separated seed derivation keeps the framework independent of RNG-specific stream-jump APIs.

## Stability Properties

This design guarantees that different domains do not consume from each other's streams once the caller creates one stream bundle for the simulation. Reordering uses of different RNGs from that bundle should not change the sequence produced by any individual domain.

This design does not guarantee that refactoring inside a single domain preserves that domain's results. If `cell-model/copy-number-segregation/v1` consumes one extra random value, the later segregation values in that same domain can change. That is expected because the domain represents one random protocol.

Repeatedly calling `make_stream` with the same domain and encoded input restarts that domain from the same seed each time. That is deterministic, but usually incorrect. Create streams once at the relevant simulation boundary.

`derive_domain_seeds([A, A]).make_streams(encoded_input)` intentionally creates two RNGs starting from the same stream seed. Duplicate domains are a protocol mistake in most simulations, but they are not a recoverable runtime error and are therefore not rejected by this API.

The full stochastic output may still change when simulation control flow changes. The goal is not to freeze all stochastic behavior, but to localize changes to the stream domains whose protocol or consumption changed.

## Relationship To Components

Simulation components should not derive seeds themselves and should not own top-level reproducibility policy. They should receive a random stream or component state that was created by the simulation layer from a `StreamDomain`.

For a copy-number segregation component, the intended usage is:

```rust
let step = StochasticStep::new_with_domain_streams(
    store,
    seed_material,
    [COPY_NUMBER_SEGREGATION_STREAM],
    |rngs, param| {
        let [segregation_rng] = rngs.as_mut();

        for index in division_indices(param) {
            divide_at(&mut cells, index, segregation_rng);
        }

        Ok(cells)
    },
    engine_factory,
);
```

This keeps simulation components focused on model behavior and keeps seed derivation centralized in `ssa-pipeline`.

## Non-Goals

This design does not attempt to provide cryptographic randomness.

This design does not require every simulation subsystem to have its own domain. Use a separate domain when the subsystem has a meaningful independent random protocol and when isolating it improves reproducibility, testing, or refactoring stability.

This design does not make persistent cache entries compatible across changes to stream domains. A domain change that affects stochastic output is a compute logic change and should use a fresh cache keyspace according to the existing `StochasticStep` keyspace contract.
