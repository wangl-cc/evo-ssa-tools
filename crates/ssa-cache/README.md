# ssa-cache

`ssa-cache` is a small execution-and-caching layer for repeated simulation workloads.

You provide compute nodes that map `input -> output` (optionally in multiple stages).
This crate focuses on the operational mechanics around that work:

- Parallel batch execution (via Rayon).
- Stable cache keys (via canonical input encoding).
- Reproducible stochastic runs (a deterministic RNG stream per repetition).
- Composable multi-stage pipelines with per-stage caches.

It does not implement SSA models for you. It helps you run your SSA code efficiently and
repeatably.

## Use Cases

Common patterns in SSA/evolutionary simulation include:

- Many repetitions of the same experiment.
- Parameter sweeps (same code, different inputs).
- A natural split between an expensive experiment stage and a downstream analysis stage.

`ssa-cache` is designed to make those workflows straightforward:

- `execute_many` consumes a Rayon `ParallelIterator` directly.
- Inputs are encoded into canonical bytes (`CanonicalEncode`) and used as cache keys.
- `StochasticStep` makes randomness reproducible and input-scoped, rather than relying on global
  RNG state.
- `Pipeline`/`.pipe(...)` expresses multi-stage compute explicitly and lets each stage own a cache.

## Core Concepts

At the lowest level, everything reduces to:

`Input` -> (canonical encode) -> `key` -> (cache hit?) -> `Output`

Key concepts:

- `Compute`: the core trait for single-item and batched execution.
- `CacheStore`: storage backend for `key -> bytes` (e.g. `HashMapStore`, or persistent `fjall`).
- `DeterministicStep`: for deterministic computations (output depends only on input).
- `StochasticStep`: for stochastic computations with reproducible per-repetition RNG streams.
  The input type is `StochasticInput { param, repetition_index }`.
- `.pipe(...)`: compose two stages; stage outputs can be cached independently.

## Quick Start

This Quick Start uses a minimal birth-death branching-process model, simulated with a Gillespie
SSA loop:

- Start with `initial_cells`.
- Run for `max_events` events.
- Each event picks either a birth (division) or a death and applies it to one cell.

The simulation kernel is written once as `birth_death_ssa<M: Marker>(...)`.
With `M = ()`, lineage tracking is disabled (cheap). With `M = LineageNode` (from `evo-marker`),
divisions carry lineage information and we can build a `PhyloTree` for downstream analysis.

### 1) Single Stochastic Compute: Trajectory

Run the experiment as a single-stage stochastic compute node without lineage tracking (`M = ()`).

### 2) Two-Stage Compute: Track Lineage Then Analyze SFS

This demonstrates a typical experiment -> analysis split:

- Stage 1 (experiment): run the stochastic simulation and produce a larger intermediate (`PhyloTree`).
- Stage 2 (analysis): compute a derived statistic from that intermediate (here: `SFS`).

Both stages can be cached. Rerunning the same inputs should avoid recomputing the intermediate and/or the analysis.

### 3) Reuse Cached Results and Stop Early with Signal

This example also demonstrates two operational behaviors you typically want in production:

- Run the same inputs twice and verify equality (the second run should be a cache hit).
- Use `ExecuteOptions::with_interrupt_signal` to stop pending parallel work; items that have not
  started return `Error::Interrupted`.

Note: the lineage/SFS portion uses `evo-marker` (`LineageNode` and `PhyloTree`). In a downstream
project, add `evo-marker` as a dependency to run that part.

```rust
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};

use evo_marker::prelude::*;
use rand::prelude::*;
use rayon::prelude::*;
use ssa_cache::prelude::*;

type Store = HashMapStore<std::collections::hash_map::RandomState>;

fn birth_death_ssa<M: Marker>(
    rng: &mut impl rand::Rng,
    initial_cells: u32,
    max_events: u32,
) -> (Vec<M>, Vec<(f64, u32)>) {
    let mut cells: Vec<M> = (0..initial_cells.max(1)).map(|_| M::default()).collect();
    let mut state = M::State::default();
    let mut t = 0.0f64;
    let mut trajectory = Vec::with_capacity(max_events as usize + 1);
    trajectory.push((t, cells.len() as u32));

    let birth_rate: f64 = 1.0;
    let death_rate: f64 = 0.1;

    for _ in 0..max_events {
        let n = cells.len();
        if n == 0 {
            break;
        }

        // Event selection: total propensities scale with the current population size.
        let n_f64 = n as f64;
        let total_birth = birth_rate * n_f64;
        let total_death = death_rate * n_f64;
        let total = total_birth + total_death;

        // Event time increment.
        let u_time: f64 = rng.random();
        let dt = -u_time.ln() / total;
        t += dt;

        let idx = rng.random_range(0..n);
        let u_event: f64 = rng.random();
        if u_event < total_birth / total {
            // Birth: divide a single cell (adds one new cell, and updates marker state if enabled).
            divide_at(&mut cells, idx, &mut state);
        } else {
            // Death: remove a single cell.
            cells.swap_remove(idx);
        }

        trajectory.push((t, cells.len() as u32));
    }

    (cells, trajectory)
}

fn main() -> ssa_cache::error::Result<()> {
    // 1) Single-stage stochastic compute: no lineage tracking (M = ()).
    let trajectory_step = StochasticStep::new(
        Store::default(),
        "branching-trajectory",
        |rng, (initial_cells, max_events): (u32, u32)| {
            let (_, trajectory) = birth_death_ssa::<()>(rng, initial_cells, max_events);
            Ok(trajectory)
        },
    );

    let trajectories = trajectory_step
        .execute_many(
            (0..10u64)
                .into_par_iter()
                .map(|rep| StochasticInput::new((50u32, 200u32), rep)),
            ExecuteOptions::default(),
        )?
        .collect::<ssa_cache::error::Result<Vec<Vec<(f64, u32)>>>>()?;
    let _ = trajectories;

    // 2) Two-stage pipeline: stage 1 builds a lineage tree, stage 2 computes SFS.
    // Stage 1 output is PhyloTree (large intermediate); stage 2 output is Vec<u32> (SFS).
    let sfs_pipeline = StochasticStep::new(
        Store::default(),
        "lineage-track",
        |rng, (initial_cells, max_events): (u32, u32)| {
            // Ensure the stage produces a valid sample for downstream analysis.
            // If the run goes extinct or ends with <2 cells, rerun with the same RNG stream.
            loop {
                let (cells, _) = birth_death_ssa::<LineageNode>(rng, initial_cells, max_events);
                if cells.len() < 2 {
                    continue;
                }
                let sample_n = cells.len().clamp(2, 64);
                let tree = PhyloTree::<2>::poisson_builder(cells, 1.0)
                    .expect("lambda must be positive")
                    .sample(rng, sample_n)
                    .build(rng);
                return Ok(tree);
            }
        },
    )
    .pipe(Store::default(), |tree: PhyloTree<2>| Ok(tree.sfs()));

    let inputs: Vec<_> = (0..8u64)
        .map(|rep| StochasticInput::new((1u32, 250u32), rep))
        .collect();
    let sfs_1 = sfs_pipeline
        .execute_many(inputs.clone().into_par_iter(), ExecuteOptions::default())?
        .collect::<ssa_cache::error::Result<Vec<Vec<u32>>>>()?;

    // 3) Run the same inputs again: results are identical (and should hit cache).
    let sfs_2 = sfs_pipeline
        .execute_many(inputs.into_par_iter(), ExecuteOptions::default())?
        .collect::<ssa_cache::error::Result<Vec<Vec<u32>>>>()?;
    assert_eq!(sfs_1, sfs_2);

    // An interrupt signal can short-circuit pending work with Error::Interrupted.
    let signal = Arc::new(AtomicBool::new(false));
    let opts = ExecuteOptions::with_interrupt_signal(signal.clone());
    signal.store(true, Ordering::Release);
    let interrupted = sfs_pipeline
        .execute_many(
            (0..8u64)
                .into_par_iter()
                .map(|rep| StochasticInput::new((1u32, 250u32), rep)),
            opts,
        )?
        .collect::<ssa_cache::error::Result<Vec<Vec<u32>>>>();
    assert!(matches!(interrupted, Err(ssa_cache::error::Error::Interrupted)));

    Ok(())
}
```

Reading the outputs:

- `trajectories`: per-repetition event-time series `Vec<(t, n_cells)>` (10 runs here).
- `sfs_1`: per-repetition SFS produced by the two-stage pipeline (8 runs here).
- `sfs_1 == sfs_2`: identical-input reruns produced identical outputs (and should benefit from
  cache hits).
- `Error::Interrupted`: external early-stop control worked.

## Cache Keyspace Contract

Caching is intentionally a pure mapping of `input-bytes -> output-bytes`.
As a result, keyspace boundaries are a caller responsibility.

- Treat each step's/stage's `cache` as its private keyspace.
- Do not reuse the same underlying keyspace across different compute logic.
- If logic, encoding, or stochastic semantics change incompatibly, use a new keyspace.

This prevents cross-logic collisions where equal input bytes accidentally return the wrong cached
output.

## Core APIs

- `Compute`: core trait (`execute` for one input, `execute_many` for batched parallel inputs).
- `ExecuteOptions`: execution controls (including `with_interrupt_signal`).
- `CodecEngine<T>`: pluggable serialization engines (e.g. `Bitcode`, `CompressedCodec<Bitcode, Lz4>` when `compress`/`lz4` are enabled).
- `DeterministicStep`: deterministic compute with an owned cache.
- `StochasticStep`: stochastic compute with reproducible per-repetition RNG streams.
- `Pipeline` / `PipelineExt`: stage composition and per-stage caching.
- `CacheStore`: cache backend interface and the in-memory implementation.

### Selecting an Engine

`DeterministicStep` / `StochasticStep` carry an engine type parameter.
By default (`bitcode` feature), `new(...)` uses `Bitcode`.
To select an explicit engine, use `new_with_engine(...)` with a typed variable:

```rust
use ssa_cache::prelude::*;

type Store = HashMapStore<std::collections::hash_map::RandomState>;
type ExplicitEngine = Bitcode;

let step: DeterministicStep<_, u64, String, ExplicitEngine, _> =
    DeterministicStep::new_with_engine(Store::default(), |x| Ok(format!("v={x}")));
```

## Feature Flags

- `bitcode` (enabled by default): `bitcode` serialization/deserialization.
- `compress` (enabled by `lz4`): framed compressed codec layer plus checksum support for custom compression engines.
- `lz4` (enabled by default): `Lz4` compression engine.
- `fjall` (disabled by default): `fjall` persistent backend.

## License

Unless otherwise specified, this crate is dual licensed under:

- **[Apache License 2.0](../../LICENSE-APACHE)**
- **[MIT License](../../LICENSE-MIT)**
