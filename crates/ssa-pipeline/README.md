# ssa-pipeline

`ssa-pipeline` is a demand-driven memoization layer for stochastic simulation and analysis pipelines.

You describe your workflow as a graph of compute nodes — stochastic simulation steps, deterministic analysis steps, or chains of both — and provide a set of inputs. The crate figures out which results are already cached, computes the rest in parallel across Rayon workers, and reuses expensive intermediate outputs automatically when multiple downstream steps depend on the same upstream result.

## Core Concepts

- `Compute` — the common trait for any node: given an input, produce an output.
- `StochasticInput<P>` — wraps a parameter value and a `repetition_index`; together they form the cache key and determine the RNG stream for that run.
- `StochasticStep` — a simulation node; the RNG for each input is seeded deterministically from `(key_material, encoded_input)`.
- `DeterministicStep` — a pure `input → output` node with no randomness, for standalone analysis or transform stages.
- `Pipeline` / `.pipe(...)` — chains an upstream node to a deterministic transform closure. Use `DeterministicStep` directly when you need a standalone deterministic node not attached to an upstream stage.
- `CacheStore` — where materialized results live: `HashMapStore` (in-memory), `Fjall2Store`, `Fjall3Store`, `RedbStore` (persistent), or `()` to disable caching.
- `CodecEngine<T>` — control how node outputs are serialized for storage; one engine instance is created per Rayon worker. See the [Codec](#codec) section for details.

## Execution Model

The flow is straightforward:

- You provide inputs — plain values for `DeterministicStep`, or `StochasticInput { param, repetition_index }` for `StochasticStep`
- You call `execute_many` on a step or pipeline
- Work fans out across Rayon worker threads; each worker holds its own encode buffer and codec engine instance
- For each input, the worker encodes it into canonical bytes and uses those bytes as the cache key; a hit returns the stored value, a miss runs the function and stores the result

`CacheStore` and `EngineFactory` stay in the background. The working model is: submit inputs, get outputs, reuse what was already computed.

## When This Crate Fits

This crate works well when:

- You want results for a specific set of inputs without precomputing everything up front
- Multiple analyses share the same upstream simulation output and you want that output computed only once
- Stochastic runs need to be reproducible and scoped to a `(param, repetition_index)` pair
- You want transparent parallelism without managing thread state yourself

It is designed as an execution and materialization layer, not a long-term result store or query engine. If your workflow leans toward exploratory data analysis, large offline collections, or table-oriented querying, pairing it with something like Parquet + DuckDB/Polars will usually serve you better.

## Quick Start

This example builds a two-stage pipeline — a stochastic simulation stage followed by a deterministic analysis stage — and runs it over eight repetitions. It demonstrates three core properties of the crate: parallel execution across Rayon workers, demand-driven caching (the second run reuses stored results without recomputing), and reproducibility (the same `(param, repetition_index)` input always produces the same output, whether computed fresh or retrieved from cache).

```rust
use rand::{Rng, RngExt};
use rayon::prelude::*;
use ssa_pipeline::prelude::*;

# #[cfg(feature = "bitcode")]
fn simulate_population(
    rng: &mut impl Rng,
    initial_cells: u32,
    steps: u32,
) -> Vec<u32> {
    let mut n = initial_cells.max(1);
    let mut trajectory = Vec::with_capacity(steps as usize + 1);
    trajectory.push(n);

    for _ in 0..steps {
        if rng.random::<f64>() < 0.6 {
            n = n.saturating_add(1);
        } else {
            n = n.saturating_sub(1);
        }
        trajectory.push(n);
    }

    trajectory
}

# #[cfg(feature = "bitcode")]
fn main() -> ssa_pipeline::error::Result<()> {
    // Stage 1: stochastic simulation.
    // Each (param, repetition_index) pair gets a deterministic RNG stream derived from
    // key_material and the encoded input, so every run produces the same trajectory for
    // the same input. Results are cached in the attached store.
    let peak_population = StochasticStep::new(
        DefaultHashMapStore::default(),   // per-stage cache
        "population-trajectory",          // key material: seeds the RNG; changing this changes all outputs
        |rng, (initial_cells, steps): (u32, u32)| Ok(simulate_population(rng, initial_cells, steps)),
        Bitcode06::default,               // engine factory: one Bitcode06 codec engine per Rayon worker
    )
    // Stage 2: deterministic analysis.
    // Chained onto stage 1 with its own cache. The cache key is the same encoded input,
    // so stage 2 results are also reused automatically on repeated calls.
    .pipe(DefaultHashMapStore::default(), |trajectory: Vec<u32>| {
        Ok(trajectory.into_iter().max().unwrap_or(0))
    });

    // Each StochasticInput pairs a parameter value with a repetition index.
    // The pair determines both the cache key and the RNG seed for that run.
    let inputs: Vec<_> = (0..8u64)
        .map(|rep| StochasticInput::new((25u32, 100u32), rep))
        .collect();

    // execute_many distributes work across Rayon workers in parallel.
    // Cache misses run the simulation and analysis; hits return stored values directly.
    let first_run = peak_population
        .execute_many(inputs.clone().into_par_iter(), ExecuteOptions::default())?
        .collect::<ssa_pipeline::error::Result<Vec<u32>>>()?;

    // Both stages hit the cache on the second run — no simulation or analysis code executes again.
    let second_run = peak_population
        .execute_many(inputs.into_par_iter(), ExecuteOptions::default())?
        .collect::<ssa_pipeline::error::Result<Vec<u32>>>()?;

    assert_eq!(first_run, second_run);
    Ok(())
}

# #[cfg(not(feature = "bitcode"))]
# fn main() {}
```

## Keyspace Contract

Memoization is a pure mapping of `input-bytes → output-bytes`. Each step or pipeline stage treats its store as a private keyspace; sharing a keyspace across different compute logic is not supported and can cause a decode error, a panic, or silently wrong results depending on how the stored bytes happen to decode against the new logic.

If compute logic, encoding, or output type changes incompatibly, use a fresh keyspace rather than reusing the old one.

## Persistent Store Wrappers

`CacheStore` is a low-level byte store. Step and pipeline types own the cache key and codec logic; the backend only controls where bytes are stored.

To use a persistent backend, open it explicitly and pass it to a step or pipeline as its dedicated store:

- `HashMapStore` — process-local, in-memory, zero setup. Best for tests, benchmarks, and short-lived runs.
- `Fjall2Store::open(keyspace, partition_name, options)`
- `Fjall3Store::open(database, keyspace_name, options)`
- `RedbStore::open(path, table_name) -> storage::Result<RedbStore>`
- `RedbStore::from_database(database, table_name) -> storage::Result<RedbStore>`
- `RedbStore::from_database_arc(database, table_name) -> storage::Result<RedbStore>`

Treat each partition, keyspace, or table as one cache namespace. If compute logic or encoding changes incompatibly, use a new namespace rather than the old one. The crate handles worker-local handle sharing internally through each store's fork behavior; you do not clone store handles yourself.

If you pair a persistent backend with a `bitcode` engine, use `Bitcode06`. Do not use the `Bitcode` alias for any persistent storage; that alias is only for in-memory or other volatile caches.

## Codec

`CodecEngine<T>` is the abstraction for serializing and deserializing node outputs. Each Rayon worker gets its own engine instance, constructed by an `EngineFactory` — any zero-argument callable that returns a fresh engine. Pass the factory as the last argument to `StochasticStep::new` or `DeterministicStep::new`.

This API is intentionally backend-agnostic. The crate can support multiple serialization backends over time, and callers may also provide their own engine implementations.

The built-in serialization backends currently available are:

- `Postcard` is the built-in serde-based backend, enabled by the `postcard` feature. Its wire format is a [published specification](https://postcard.jamesmunns.com/wire-format), stable within the same major version; a breaking wire-format change requires a new `postcard` major version.
- `Bitcode06` explicitly names the `bitcode v0.6` backend. Suitable for both persistent and volatile caches; treat any future `bitcode` major-version upgrade as a data migration step.
- `Bitcode` is a convenience alias for the latest `bitcode` backend (currently `Bitcode06`). **Do not** use for persistent storage — it silently tracks the latest version.

Rule of thumb:

- Use `Postcard` for persistent storage if your types already derive `serde` traits. Minor/patch upgrades of `postcard` are always wire-compatible; only a major version bump would require migration.
- Use `Bitcode06` for persistent storage when throughput matters. Treat any future `bitcode` major version upgrade as a data migration step.
- Never use `Bitcode` for persistent storage; it silently follows the latest `bitcode` backend.

### Compressed Codec

For workloads where storage size matters, `CompressedCodec<E, C, P>` wraps any existing engine with a framed compression layer. The `lz4` and `zstd` features provide ready-made compression algorithms. You can also plug in a custom `CompressPolicy` to decide at runtime whether compression is worth applying for a given payload:

```rust
# #[cfg(all(feature = "lz4", feature = "bitcode"))]
# {
use ssa_pipeline::{
    cache::codec::compress::policy::{CompressionAction, CompressPolicy},
    prelude::*,
};

struct TunedPolicy;

impl CompressPolicy for TunedPolicy {
    fn before_compress(&self, raw_size: usize) -> CompressionAction {
        if raw_size < 8 * 1024 {
            CompressionAction::Raw
        } else {
            CompressionAction::Compress
        }
    }

    fn after_compress(
        &self,
        raw_frame_size: usize,
        compressed_frame_size: usize,
    ) -> CompressionAction {
        if raw_frame_size.saturating_sub(compressed_frame_size) >= 512 {
            CompressionAction::Compress
        } else {
            CompressionAction::Raw
        }
    }
}

let _engine = CompressedCodec::<Bitcode06, Lz4>::new(Bitcode06::default()).with_policy(TunedPolicy);
# }
```

`.with_max_len(...)` sets a size limit at both encode time (serialized payload) and decode time (compressed payload); pass `0` to remove it. The decode-time check does not apply to raw frames and is independent of `CompressPolicy`.

## Feature Flags

- `bitcode` (disabled by default): `bitcode` serialization/deserialization via `Bitcode` / `Bitcode06`.
- `compress` (automatically enabled by `lz4`/`zstd`; can also be enabled directly for custom engines): framed compressed codec layer and checksum support.
- `lz4` (disabled by default): `Lz4` compression engine.
- `postcard` (disabled by default): `postcard` + `serde` serialization/deserialization via `Postcard`.
- `zstd` (disabled by default): `Zstd` compression engine with runtime-configurable compression level.
- `fjall2` (disabled by default): Fjall v2 persistent backend wrapper.
- `fjall3` (disabled by default): Fjall v3 persistent backend wrapper.
- `redb` (disabled by default): `redb` persistent backend wrapper.

## License

Unless otherwise specified, this crate is dual licensed under:

- **[Apache License 2.0](../../LICENSE-APACHE)**
- **[MIT License](../../LICENSE-MIT)**
