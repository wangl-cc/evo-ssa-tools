# Random Streams Design

This document describes the design for reproducible named random streams in `ssa-pipeline`.

## Motivation

`StochasticStep::new` gives each input one deterministic `Xoshiro256PlusPlus` stream. That model is simple and correct for simulations where randomness is naturally consumed as one sequential trajectory.

Some simulations need stronger engineering stability. A model may contain several random subsystems such as event scheduling, mutation sampling, copy-number segregation, and final sampling. If all of them consume one shared stream, changing the implementation of one subsystem can shift the random values consumed by later subsystems. The result is still reproducible for the exact same code, but it is sensitive to code ordering and local refactors.

The long-term goal is module-level stream isolation with stream derivation centralized at the simulation boundary.

## Core Model

Random stream identity has three explicit axes:

```text
ExperimentDomain
StreamDomain
StochasticInput { param, repetition_index }
```

`ExperimentDomain` is the stable namespace for one stochastic experiment or model protocol. Use a versioned, static name such as `experiment/cell-copy-number/v1`.

`StreamDomain` is the stable namespace for one random subsystem protocol. Examples include `cell-model/division-event/v1`, `cell-model/copy-number-segregation/v1`, or `cell-model/mutation-sampling/v1`.

`StochasticInput` identifies the concrete run. Its `param` is the model input and its `repetition_index` is the replicate index.

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

The experiment domain and stream domain are the public protocol namespaces for random stream derivation.

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

The generic seed type records whether the step stores a single domain seed or a fixed bundle of domain seeds.

## Domain Names

Define experiment and stream domains as constants:

```rust
pub const EXPERIMENT: ExperimentDomain =
    ExperimentDomain::new("experiment/ssa-with-copy-number/v1");
pub const SEGREGATION_STREAM: StreamDomain =
    StreamDomain::new("model/copy-number-segregation/v1");
```

Domain names should include the owning crate or system, the stochastic protocol, and a version. Use a new domain version when changing the random protocol for an experiment or stream.

Use `StochasticInput::repetition_index` for replicate indices:

```rust
let inputs = (0..128u64)
    .map(|rep| StochasticInput::new(model_param, rep));
```

## Seed Derivation

Streams are derived using deterministic domain separation:

```text
domain_seed = derive_key(stream_domain, experiment_domain_bytes)
domain_stream_seed = keyed_hash(domain_seed, encoded_input_bytes)
domain_stream_rng = Xoshiro256PlusPlus::from_seed(domain_stream_seed)
```

For single-stream execution, `stream_domain` is the crate-internal single-stream domain.

The domain `cell-model/copy-number-segregation/v1` maps to the same domain seed regardless of whether other domains are requested before or after it. Domain-separated seed derivation keeps stream construction independent of RNG-specific stream-jump APIs.

## Stability Properties

This design guarantees that different domains do not consume from each other's streams once the caller creates one stream bundle for the simulation. Reordering uses of different RNGs from that bundle should not change the sequence produced by any individual domain.

Within a single domain, random values are still consumed sequentially. If `cell-model/copy-number-segregation/v1` consumes one extra random value, later segregation values in that domain can change.

Repeatedly calling `make_stream` with the same domain seed and encoded input restarts that domain from the same seed each time. That is deterministic, but usually incorrect. Create streams once at the relevant simulation boundary.

`derive_domain_seeds([A, A]).make_streams(encoded_input)` creates two RNGs starting from the same stream seed. Duplicate domains produce duplicate streams.

The full stochastic output may still change when simulation control flow changes. The goal is not to freeze all stochastic behavior, but to localize changes to the stream domains whose protocol or consumption changed.

## Relationship To Components

Simulation components receive random streams or component state created by the simulation layer from an `ExperimentDomain` and `StreamDomain`.

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

The streams are intended for reproducible simulation, not cryptographic randomness.

Use a separate stream domain when a subsystem has a meaningful independent random protocol and when isolating it improves reproducibility, testing, or refactoring stability.

This design does not make persistent cache entries compatible across changes to experiment domains or stream domains. A domain change that affects stochastic output is a compute logic change and should use a fresh cache keyspace according to the existing `StochasticStep` keyspace contract.
