# ssa-workflow

`ssa-workflow` is a demand-driven memoization layer for stochastic simulation and analysis workflows.

You describe a workflow as ordinary Rust tasks and transforms, provide inputs, and collect outputs. The crate computes cache misses in parallel, reuses cached results, and keeps stochastic runs reproducible by deriving RNG streams from stable computation identity plus canonical input bytes.

## When This Crate Fits

Use `ssa-workflow` when the workflow is driven by the parameter sets you actually ask for. Instead of splitting work into a large batch-generation phase and a later analysis phase, you define simulations and analyses as linked tasks and transforms, submit the requested inputs, and let the crate compute only the missing intermediate and final results.

This fits workflows where:

- analyses pull on the simulation outputs they need, rather than scanning a precomputed result table
- parameter sweeps are selective, iterative, or shared by several downstream analyses
- multiple analyses may depend on the same expensive upstream simulation output
- stochastic runs must be reproducible for each `(parameter, repetition_index)` pair
- you want transparent parallel execution and memoization while keeping workflow code in Rust

It is not a replacement for table-oriented analytics. If your workflow is “run a large batch, write all results, then explore/query them later”, a storage-first setup such as Parquet plus DuckDB or Polars may be a better fit. `ssa-workflow` is for demand-driven execution: provide the parameter set you need now, compute what is missing, and reuse what was already materialized.

## Quick Start

This example builds a two-stage workflow: a stochastic simulation followed by a deterministic analysis. The first run computes and caches both stages; the second run reuses cached results.

```rust
use rand::{Rng, RngExt};
use ssa_workflow::prelude::*;

fn simulate_population(
    rng: &mut impl Rng,
    initial_cells: u32,
    max_events: u32,
) -> Vec<(f64, u32)> {
    let birth_rate = 0.8;
    let death_rate = 0.4;
    let mut cells = initial_cells.max(1);
    let mut time = 0.0;
    let mut trajectory = Vec::with_capacity(max_events as usize + 1);
    trajectory.push((time, cells));

    for _ in 0..max_events {
        let birth_propensity = birth_rate * cells as f64;
        let death_propensity = death_rate * cells as f64;
        let total_propensity = birth_propensity + death_propensity;
        if total_propensity == 0.0 {
            break;
        }

        let u = rng.random::<f64>().clamp(f64::MIN_POSITIVE, 1.0);
        time += -u.ln() / total_propensity;

        let reaction_threshold = rng.random::<f64>() * total_propensity;
        if reaction_threshold < birth_propensity {
            cells = cells.saturating_add(1);
        } else {
            cells = cells.saturating_sub(1);
        }
        trajectory.push((time, cells));
    }

    trajectory
}

fn main() -> ssa_workflow::error::Result<()> {
    // Stable computation id: change it when the simulation semantics change.
    let trajectory = StochasticTask::builder("birth-death-ssa-trajectory-v1")
        .function(|rng, (initial_cells, max_events): (u32, u32)| {
            Ok(simulate_population(rng, initial_cells, max_events))
        })
        // Process-local cache for trajectory outputs.
        .cache(ManagedHashCache::<Vec<(f64, u32)>>::default())
        .build()?;

    // Dependent analysis: its identity includes the upstream trajectory path.
    let peak_population = trajectory
        .transform("peak-population-v1")
        .function(|trajectory: Vec<(f64, u32)>| {
            Ok(trajectory
                .into_iter()
                .map(|(_, cells)| cells)
                .max()
                .unwrap_or(0))
        })
        .cache(ManagedHashCache::<u32>::default())
        .build()?;

    // Same parameter set, eight independent stochastic repetitions.
    let inputs: Vec<_> = (0..8u64)
        .map(|rep| StochasticInput::new((25u32, 100u32), rep))
        .collect();

    // Execution is demand-driven: cache misses compute, hits reuse stored results.
    let first_run = peak_population.with_inputs(inputs.clone()).collect()?;
    let second_run = peak_population.with_inputs(inputs).collect()?;

    assert_eq!(first_run, second_run);
    Ok(())
}
```

## Core Concepts

- `StochasticTask` runs a simulation with reproducible RNG streams. Each `StochasticInput<P>` combines a parameter value with a `repetition_index`.
- `DeterministicTask` runs a pure `input -> output` root computation.
- `.transform(...)` builds a dependent deterministic analysis from an upstream task or transform.
- `.stochastic_transform(...)` builds a dependent stochastic analysis from an upstream result.
- Task and transform ids are stable, versioned names for semantic results, such as `birth-death-ssa-trajectory-v1`.
- Cache providers such as `ManagedHashCache`, `ManagedLruCache`, and `PersistentCacheProvider` attach storage to a task or transform before it is built.

## Computation Identity

Every task and transform needs a stable id. That id becomes part of the computation path used for caching, reproducible RNG streams, and downstream transform identity.

Use short, versioned, backend-safe names:

```rust
# use ssa_workflow::prelude::*;
# fn main() -> ssa_workflow::error::Result<()> {
let task = DeterministicTask::builder("summary-statistics-v1")
    .function(|x: u32| Ok(x + 1))
    .cache(ManagedHashCache::<u32>::default())
    .build()?;
# let _ = task;
# Ok(())
# }
```

Identifiers use stable segments with ASCII letters, digits, and `-`; `_` is reserved for derived cache names. Prefer lowercase kebab-case and bump the version when the meaning of a result changes. Persistent namespace names use `_` between path or codec segments and `__` between the computation path and codec format.

Dependent computation paths render from the current result back to their roots. For example, a transform id `summary-v1` built from a task id `trajectory-v1` renders as `summary-v1_trajectory-v1`, read as "summary of trajectory". Persistent namespaces and RNG seed material use this same segment order.

If you change compute logic, output type, or persistent encoding incompatibly, use a new computation id or codec format so old cached bytes are not reused as a different result.

## Transforms

Transforms are dependent compute nodes. A transform's computation path includes the upstream path plus the transform's own id, so changing an upstream task automatically selects a different cache space for downstream results.

Use `.transform(...)` when the downstream computation only depends on the upstream output:

```rust
# use ssa_workflow::prelude::*;
# let trajectory = DeterministicTask::builder("birth-death-ssa-trajectory-v1")
    .function(|input: u32| Ok(vec![input]))
#     .cache(ManagedHashCache::<Vec<u32>>::default())
#     .build()?;
let peak = trajectory
    .transform("peak-population-v1")
    .function(|trajectory: Vec<u32>| {
        Ok(trajectory.into_iter().max().unwrap_or(0))
    })
    .cache(ManagedHashCache::<u32>::default())
    .build()?;
# let _ = peak;
# Ok::<(), ssa_workflow::error::Error>(())
```

Use `.transform("...").function_with_param(...)` when an analysis parameter is part of the result semantics. Use `.stochastic_transform("...")` when downstream analysis samples from an upstream result.

Random variable names used with `.streams(...)` are opaque RNG seed labels. They do not become storage namespace segments and are not restricted like task or transform ids. Keep them stable for reproducibility; changing a label changes only that variable's derived stream.

## Choosing a Cache

Start with an in-memory managed cache unless you need persistence across process runs.

- `ManagedHashCache<T>`: unbounded in-memory cache provider; always available.
- `ManagedLruCache<T>`: bounded in-memory cache provider with LRU eviction; requires the `lru` feature.
- `PersistentCacheProvider<SP, CE>`: persistent cache provider composed by calling `.with_codec(...)` on a storage provider.
- `()`: disables caching; every input is recomputed.

Unbounded in-memory caching:

```rust
use ssa_workflow::prelude::*;

let _cache = ManagedHashCache::<String>::default();
```

Bounded in-memory caching:

```rust
# #[cfg(feature = "lru")]
# {
use std::num::NonZeroUsize;
use ssa_workflow::prelude::*;

let _cache = ManagedLruCache::<String>::new(NonZeroUsize::new(256).expect("capacity is non-zero"));
# }
```

## Persistent Caches

Persistent caches combine a storage provider with a codec. The storage provider decides where bytes live; the codec decides how typed outputs are encoded.

```no_run
use ssa_workflow::prelude::*;

# #[cfg(all(feature = "fjall3", feature = "bitcode06"))]
# {
let db = fjall3::Database::builder("./my-cache").open().unwrap();
let cache = Fjall3StorageProvider::new(db)
    .with_codec(Bitcode06::default());

let task = DeterministicTask::builder("expensive-analysis-v1")
    .function(|x: u32| Ok(x * 2))
    .cache(cache)
    .build()?;
# let _ = task;
# }
# Ok::<(), ssa_workflow::error::Error>(())
```

Use `.with_keyspace_options(...)` before `.with_codec(...)` when you need custom Fjall keyspace creation options. Create multiple `Fjall3StorageProvider`s from the same database when different cache families need different options. The provider opens one physical namespace per computation path and codec format.

## Codecs

Persistent caches need a `CodecEngine<T>` for the output type `T`. Each codec has a stable `VALUE_FORMAT`; the value format is included in the persistent namespace so incompatible encodings do not share storage.

Built-in serialization choices:

- `Bitcode06` (`bitcode06` feature): fast binary encoding tied to `bitcode` 0.6.
- `Postcard` (`postcard` feature): compact serde-based encoding with a published wire spec.

For workloads where storage size matters, wrap a codec with framed compression:

```rust
# #[cfg(all(feature = "lz4", feature = "bitcode06"))]
# {
use ssa_workflow::prelude::*;

let _engine = Bitcode06::default()
    .compress(Lz4)
    .with_max_len(64 * 1024 * 1024)
    .build();
# }
```

Use `Lz4` when speed matters. Use `Zstd` when stronger compression is worth the extra CPU cost.

## Feature Flags

- `bitcode06` (disabled by default): `bitcode` 0.6 serialization/deserialization via `Bitcode06`.
- `compress` (enabled by `lz4` or `zstd`): framed compression layer and checksum support.
- `fjall3` (disabled by default): Fjall v3 storage provider.
- `lru` (disabled by default): bounded in-memory cache with LRU eviction.
- `lz4` (disabled by default): `Lz4` compression.
- `postcard` (disabled by default): `postcard` + `serde` serialization/deserialization via `Postcard`.
- `zstd` (disabled by default): `Zstd` compression with runtime-configurable compression level.

## License

Unless otherwise specified, this crate is dual licensed under:

- **[Apache License 2.0](../../LICENSE-APACHE)**
- **[MIT License](../../LICENSE-MIT)**
