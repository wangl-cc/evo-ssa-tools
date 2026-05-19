# Random Streams Design

This document describes the design for reproducible named random streams in `ssa-pipeline`.

## Motivation

`StochasticStep::new` gives each input one deterministic `Xoshiro256PlusPlus` stream. That model is simple and correct for simulations where randomness is naturally consumed as one sequential trajectory.

Some simulations need stronger engineering stability. A model may contain several random subsystems such as event scheduling, mutation sampling, copy-number segregation, and final sampling. If all of them consume one shared stream, changing the implementation of one subsystem can shift the random values consumed by later subsystems. The result is still reproducible for the exact same code, but it is sensitive to code ordering and local refactors.

The long-term goal is to support module-level stream isolation without making random seed management the responsibility of each simulation component.

## Core Model

Random stream identity has three explicit axes:

```text
ExperimentDomain
StreamDomain
StochasticInput { param, repetition_index }
```

`ExperimentDomain` is the stable namespace for one stochastic experiment or model protocol. It is not a per-run random seed. Use a versioned, static name such as `experiment/cell-copy-number/v1`. If the stochastic protocol changes in a way that should invalidate reproducibility expectations, use a new experiment domain version.

`StreamDomain` is the stable namespace for one random subsystem protocol. Examples include `cell-model/division-event/v1`, `cell-model/copy-number-segregation/v1`, or `cell-model/mutation-sampling/v1`.

`StochasticInput` identifies the concrete run. Its `param` is the model input and its `repetition_index` is the replicate index. Different random trajectories for the same experiment and parameter should use different repetition indices, not ad hoc seed strings.

Seed derivation is:

```text
ExperimentDomain + StreamDomain
-> DomainSeed

DomainSeed + encoded_input_bytes
-> Xoshiro256PlusPlus
```

The single-stream constructor uses the same model with an internal stream domain:

```text
ExperimentDomain + internal single-stream domain
-> DomainSeed

DomainSeed + encoded_input_bytes
-> Xoshiro256PlusPlus
```

There is intentionally no public root seed. The experiment domain and stream domain are the two protocol namespaces users should reason about.

## Seed Types

Seed handling lives under `stochastic::seed`.

```rust
pub struct ExperimentDomain(&'static str);
pub struct StreamDomain(&'static str);
pub struct DomainSeed {
    // private fields
}
pub struct DomainSeeds<const N: usize> {
    // private fields
}

impl ExperimentDomain {
    pub const fn new(name: &'static str) -> Self;
    pub const fn as_str(self) -> &'static str;
    pub fn derive_single_stream_seed(self) -> DomainSeed;
    pub fn derive_domain_seed(self, domain: StreamDomain) -> DomainSeed;
    pub fn derive_domain_seeds<const N: usize>(self, domains: [StreamDomain; N]) -> DomainSeeds<N>;
}

impl DomainSeed {
    pub fn make_stream(&self, encoded_input: &[u8]) -> Xoshiro256PlusPlus;
}

impl<const N: usize> DomainSeeds<N> {
    pub fn make_streams(&self, encoded_input: &[u8]) -> StochasticStreams<N>;
}
```

`DomainSeed` and `DomainSeeds<N>` are opaque and do not expose raw seed bytes. `derive_domain_seed` and `derive_domain_seeds` are named as derivation operations because they create domain-specific seeds rather than fetching stored domain objects.

## StochasticStep API

Define experiment and stream domains as constants:

```rust
pub const CELL_COPY_NUMBER_EXPERIMENT: ExperimentDomain =
    ExperimentDomain::new("experiment/cell-copy-number/v1");

pub const DIVISION_EVENT_STREAM: StreamDomain =
    StreamDomain::new("cell-model/division-event/v1");

pub const COPY_NUMBER_SEGREGATION_STREAM: StreamDomain =
    StreamDomain::new("cell-model/copy-number-segregation/v1");
```

The single-stream constructor is:

```rust
let step = StochasticStep::new(
    store,
    CELL_COPY_NUMBER_EXPERIMENT,
    |rng, param| {
        // simulation logic using one stream
        Ok(output)
    },
    engine_factory,
);
```

This derives one `DomainSeed` from the experiment domain and the crate-internal single-stream domain, then creates the input-scoped RNG from that domain seed and the encoded input.

The domain-stream constructor is:

```rust
let step = StochasticStep::new_with_domain_streams(
    store,
    CELL_COPY_NUMBER_EXPERIMENT,
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

The domain-stream constructor derives `DomainSeeds<N>` once at construction and creates input-scoped `StochasticStreams<N>` during execution.

Internally, both modes share the same `StochasticStep` execution shell, but the stored seed type differs:

```rust
StochasticStep<C, P, O, DomainSeed, F, EF>
StochasticStep<C, P, O, DomainSeeds<N>, F, EF>
```

The field is intentionally named `seed`. Whether it is a single domain seed or a fixed bundle of domain seeds is expressed by the generic type.

## Domain Names

Experiment and stream domains should be constants, not ad hoc strings spread through simulation code:

```rust
pub const EXPERIMENT: ExperimentDomain =
    ExperimentDomain::new("experiment/ssa-with-copy-number/v1");
pub const SEGREGATION_STREAM: StreamDomain =
    StreamDomain::new("model/copy-number-segregation/v1");
```

Domain names should include the owning crate or system, the stochastic protocol, and a version. Changing the random protocol should use a new domain version instead of silently changing the meaning of an existing domain.

The repetition axis belongs in `StochasticInput`, not in the experiment domain:

```rust
let inputs = (0..128u64)
    .map(|rep| StochasticInput::new(model_param, rep));
```

Do not use `ExperimentDomain::new("experiment/foo/run-17")` to get another random trajectory. That creates another experiment namespace rather than another replicate of the same experiment.

## Seed Derivation

Streams are derived using deterministic domain separation:

```text
domain_seed = derive_key(stream_domain, experiment_domain_bytes)
domain_stream_seed = keyed_hash(domain_seed, encoded_input_bytes)
domain_stream_rng = Xoshiro256PlusPlus::from_seed(domain_stream_seed)
```

For single-stream execution, `stream_domain` is the crate-internal single-stream domain.

This is intentionally seed derivation, not RNG sampling. The domain `cell-model/copy-number-segregation/v1` maps to the same domain seed regardless of whether other domains are requested before or after it.

This also means `SeedableRng::fork` is not the right core primitive for domain streams. `fork` is useful for creating another RNG from a current RNG state, but it is order-sensitive. Adding a new fork before an existing fork changes the later child stream.

`jump` and `long_jump` are also not the right general abstraction for this crate. They are useful when a chosen RNG exposes them, but the `rand` crate's exported `Xoshiro256PlusPlus` does not expose a generic stream-splitting trait. Domain-separated seed derivation keeps the framework independent of RNG-specific stream-jump APIs.

## Stability Properties

This design guarantees that different domains do not consume from each other's streams once the caller creates one stream bundle for the simulation. Reordering uses of different RNGs from that bundle should not change the sequence produced by any individual domain.

This design does not guarantee that refactoring inside a single domain preserves that domain's results. If `cell-model/copy-number-segregation/v1` consumes one extra random value, the later segregation values in that same domain can change. That is expected because the domain represents one random protocol.

Repeatedly calling `make_stream` with the same domain seed and encoded input restarts that domain from the same seed each time. That is deterministic, but usually incorrect. Create streams once at the relevant simulation boundary.

`derive_domain_seeds([A, A]).make_streams(encoded_input)` intentionally creates two RNGs starting from the same stream seed. Duplicate domains are a protocol mistake in most simulations, but they are not a recoverable runtime error and are therefore not rejected by this API.

The full stochastic output may still change when simulation control flow changes. The goal is not to freeze all stochastic behavior, but to localize changes to the stream domains whose protocol or consumption changed.

## Relationship To Components

Simulation components should not derive seeds themselves and should not own top-level reproducibility policy. They should receive a random stream or component state that was created by the simulation layer from an `ExperimentDomain` and `StreamDomain`.

For a copy-number segregation component, the intended usage is:

```rust
let step = StochasticStep::new_with_domain_streams(
    store,
    EXPERIMENT,
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

This keeps simulation components focused on model behavior and keeps stream derivation centralized in `ssa-pipeline`.

## Non-Goals

This design does not attempt to provide cryptographic randomness.

This design does not require every simulation subsystem to have its own domain. Use a separate domain when the subsystem has a meaningful independent random protocol and when isolating it improves reproducibility, testing, or refactoring stability.

This design does not make persistent cache entries compatible across changes to experiment domains or stream domains. A domain change that affects stochastic output is a compute logic change and should use a fresh cache keyspace according to the existing `StochasticStep` keyspace contract.
