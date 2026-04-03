# ssa-pipeline

`ssa-pipeline` is a demand-driven memoization layer for stochastic simulation and analysis pipelines.

You describe your workflow as a graph of compute nodes — stochastic simulation steps, deterministic analysis steps, or chains of both — and provide a set of inputs. The crate figures out which results are already cached, computes the rest in parallel across workers, and reuses expensive intermediate outputs automatically when multiple downstream steps depend on the same upstream result.

## Core Concepts

- `Compute` — the common trait for any node: given an input, produce an output.
- `StochasticInput<P>` — wraps a parameter value and a `repetition_index`; together they form the cache key and determine the RNG stream for that run.
- `StochasticStep` — a simulation node; the RNG for each input is seeded deterministically from `(key_material, encoded_input)`.
- `DeterministicStep` — a pure `input → output` node with no randomness, for standalone analysis or transform stages.
- `Pipeline` / `.pipe(...)` — chains an upstream node to a deterministic transform closure. Use `DeterministicStep` directly when you need a standalone deterministic node not attached to an upstream stage.
- `Cache<T>` — the execution-facing cache abstraction used by steps and pipelines. It can be backed by `HashObjectCache<T>` (unbounded in-memory), `LruObjectCache<T>` (bounded with LRU eviction, requires `lru` feature), `EncodedCache<S, CE>` (raw store + codec for persistent storage), or `()` to disable caching.
- `CacheStore` — the low-level `key -> encoded bytes` storage abstraction behind raw backends like `Fjall2Store`, `Fjall3Store`, and `RedbStore`.
- `CodecEngine<T>` — controls how node outputs are serialized for persistent storage. Each worker gets its own engine instance, obtained via `Fork::fork`.

## Execution Model

The flow is straightforward:

- You provide inputs — plain values for `DeterministicStep`, or `StochasticInput { param, repetition_index }` for `StochasticStep`
- You call `execute_many` on a step or pipeline
- Work fans out across worker threads; each worker holds its own input-encoding buffer and its own cloned execution state
- For each input, the worker encodes it into canonical bytes and uses those bytes as the cache key; a hit returns the stored value, a miss runs the function and stores the result
- If the cache is an `EncodedCache<S, CE>`, each worker also gets its own codec engine instance; typed object caches such as `HashObjectCache<T>` and `LruObjectCache<T>` do not involve a codec engine at all

The working model is: submit inputs, get outputs, reuse what was already computed.

## When This Crate Fits

This crate works well when:

- You want results for a specific set of inputs without precomputing everything up front
- Multiple analyses share the same upstream simulation output and you want that output computed only once
- Stochastic runs need to be reproducible and scoped to a `(param, repetition_index)` pair
- You want transparent parallelism without managing thread state yourself

It is designed as an execution and materialization layer, not a long-term result store or query engine. If your workflow leans toward exploratory data analysis, large offline collections, or table-oriented querying, pairing it with something like Parquet + DuckDB/Polars will usually serve you better.

## Quick Start

This example builds a two-stage pipeline — a stochastic simulation stage followed by a deterministic analysis stage — and runs it over eight repetitions. It demonstrates three core properties of the crate: parallel execution across workers, demand-driven caching (the second run reuses stored results without recomputing), and reproducibility (the same `(param, repetition_index)` input always produces the same output, whether computed fresh or retrieved from cache).

```rust
use rand::{Rng, RngExt};
use rayon::prelude::*;
use ssa_pipeline::prelude::*;

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

fn main() -> ssa_pipeline::error::Result<()> {
    // Stage 1: stochastic simulation.
    // Each (param, repetition_index) pair gets a deterministic RNG stream derived from
    // key_material and the encoded input, so every run produces the same trajectory for
    // the same input. Results are cached in memory.
    let peak_population = StochasticStep::new(
        DefaultHashObjectCache::default(),
        "population-trajectory",          // key material: seeds the RNG; changing this changes all outputs
        |rng, (initial_cells, steps): (u32, u32)| Ok(simulate_population(rng, initial_cells, steps)),
    )
    // Stage 2: deterministic analysis.
    // Chained onto stage 1 with its own cache. The cache key is the same encoded input,
    // so stage 2 results are also reused automatically on repeated calls.
    .pipe(DefaultHashObjectCache::default(), |trajectory: Vec<u32>| {
        Ok(trajectory.into_iter().max().unwrap_or(0))
    });

    // Each StochasticInput pairs a parameter value with a repetition index.
    // The pair determines both the cache key and the RNG seed for that run.
    let inputs: Vec<_> = (0..8u64)
        .map(|rep| StochasticInput::new((25u32, 100u32), rep))
        .collect();

    // execute_many distributes work across workers in parallel.
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
```

## Keyspace Contract

Memoization is a pure mapping of `input-bytes → output-bytes` at the raw storage layer. Each step or pipeline stage treats its cache as a private keyspace; sharing a keyspace across different compute logic is not supported and can cause a decode error, a panic, or silently wrong results depending on how the stored bytes happen to decode against the new logic.

If compute logic, encoding, or output type changes incompatibly, use a fresh keyspace rather than reusing the old one.

## Cache Choices

Use these cache choices depending on what you need:

- `HashObjectCache<T>` — process-local unbounded typed object cache. Zero extra dependencies; the default for tests, benchmarks, and short-lived runs. Fork clones the underlying `Arc` so all workers share the same map.
- `LruObjectCache<T>` (requires `lru` feature) — process-local bounded typed object cache with LRU eviction. Best when you need a cap on memory usage.
- `EncodedCache<S, CE>` — wraps any `CacheStore` with a `CodecEngine` for persistent or encoded-byte storage. Pairs with `Fjall2Store`, `Fjall3Store`, or `RedbStore`.
- `()` — disables caching entirely; every input is always recomputed.

To use a persistent backend, open it explicitly and wrap it with `EncodedCache`:

```no_run
use ssa_pipeline::prelude::*;
# #[cfg(feature = "fjall3")]
# {
// Open an existing Fjall v3 database and create an encoded cache from it.
// let db = fjall3::Database::builder("./my-db").open().unwrap();
// let store = Fjall3Store::open(db, "my-step", None).unwrap();
// let cache = EncodedCache::new(store, Bitcode06::default());
# }
```

Available persistent store constructors:

- `Fjall2Store::open(keyspace, partition_name, options)` (requires `fjall2` feature)
- `Fjall3Store::open(database, keyspace_name, options)` (requires `fjall3` feature)
- `RedbStore::open(path, table_name)` (requires `redb` feature)
- `RedbStore::from_database(database, table_name)`
- `RedbStore::from_database_arc(database, table_name)`

Treat each partition, keyspace, or table as one cache namespace. If compute logic or encoding changes incompatibly, use a new namespace rather than the old one. The crate handles worker-local handle sharing internally through each store's `Fork` behavior; you do not clone store handles yourself.

For unbounded typed in-memory caching (always available):

```rust
use ssa_pipeline::prelude::*;
let _cache = DefaultHashObjectCache::<String>::default();
```

For bounded typed in-memory caching with LRU eviction (requires `lru` feature):

```rust
# #[cfg(feature = "lru")]
# {
use std::num::NonZeroUsize;
use ssa_pipeline::prelude::*;

let _cache = LruObjectCache::<String>::new(NonZeroUsize::new(256).expect("capacity is non-zero"));
# }
```

## Codec

`CodecEngine<T>` is the abstraction for serializing and deserializing node outputs for persistent storage. `EncodedCache<S, CE>` owns one engine per worker; when the cache is forked for a new worker, the engine's `Fork` implementation produces a fresh instance with the same configuration.

The built-in serialization backends currently available are:

- `Postcard` is the built-in serde-based backend, enabled by the `postcard` feature. Its wire format is a [published specification](https://postcard.jamesmunns.com/wire-format), stable within the same major version; a breaking wire-format change requires a new `postcard` major version.
- `Bitcode06` explicitly names the `bitcode v0.6` backend. Treat any future `bitcode` major-version upgrade as a data migration step.

Rule of thumb:

- Use `Postcard` when your types already derive `serde` traits. Minor/patch upgrades of `postcard` are always wire-compatible; only a major version bump would require migration.
- Use `Bitcode06` when throughput matters. Treat any future `bitcode` major version upgrade as a data migration step.

### Compressed Codec

For workloads where storage size matters, `CompressedCodec<E, C, P>` wraps any existing engine with a framed compression layer. The `lz4` and `zstd` features provide ready-made compression algorithms. You can also plug in a custom `CompressPolicy` to decide at runtime whether compression is worth applying for a given payload:

```rust
# #[cfg(all(feature = "lz4", feature = "bitcode06"))]
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

- `bitcode06` (disabled by default): `bitcode v0.6` serialization/deserialization via `Bitcode06`.
- `compress` (automatically enabled by `lz4`/`zstd`; can also be enabled directly for custom engines): framed compressed codec layer and checksum support.
- `lru` (disabled by default): bounded in-memory typed cache with LRU eviction via `LruObjectCache<T>`.
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
