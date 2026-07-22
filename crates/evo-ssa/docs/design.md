# evo-ssa Design and Implementation

## Scope

`evo-ssa` targets evolutionary systems with a fixed set of reaction families and a dynamic, potentially very large set of concrete channels inside each family. The implemented methods are exact when propensities remain constant between reported updates. Exogenous changes must be represented as events; continuously time-dependent hazards need a separate nonhomogeneous clock contract.

## Ownership

The model owns biological state, propensity formulas, reaction effects, event meaning, and dependency knowledge. Scheduling algorithms own cached propensities or NRM clocks and channel selection. `Simulation` and `FamilySimulation` own model state, simulation time, run termination, error state, and random-stream routing.

The framework does not infer a global dependency graph. `fire` mutates state and returns a compact event. `refresh_after_event` follows the model's event-local dependency edges and emits affected channel slots through `ChannelRecomputeSink`. Keeping mutation and dependency projection separate gives every family scheduler the same event and observer boundary without allocating a generic state-delta object.

## Model APIs

`EvolvingModel` is the flexible API. Concrete channels have arbitrary hashable keys and cloneable payloads. `ChannelEditor` emits `Upsert`, `Recompute`, and `Remove`, and `DirectScheduler` maps active keys onto reusable segment-tree slots.

`StaticFamilyModel` plus `StaticReactionFamily` is the performance API. `reaction_families!` generates an arbitrary-length family-list type and a named enum whose conversions produce `FamilyId<F>`. The bundle brand prevents a family id from one model's family set from being passed to another model's update sink. `family_list!` builds the recursively composed value, so there is no tuple-arity limit or family-count-specific scheduler implementation.

## Family Channel Lifecycle

For a running static-family simulation:

- Family order is fixed by the declared family-list type.
- Local channel ids are dense scheduler slots.
- `channel_count` may append channels or remove a suffix.
- Existing slots retain their biological meaning until explicitly reinitialized.
- A removed interior channel may remain at zero propensity and later be reused with `reinitialize` or `reinitialize_family`.
- Newly appended suffix channels are initialized automatically and do not require dependency updates.

`Recompute` preserves a non-fired NRM channel's residual hazard. `Reinitialize` discards the previous slot identity and draws a fresh clock. This distinction lets a model keep dense storage near its peak active-slot count instead of retaining one scheduler slot for every species ever created.

## Update Validation

Cross-family updates use named `FamilyId<F>` values rather than positional integers. Every reported local slot is validated after structural resizing. A recompute of a missing dynamic key or out-of-range family-local slot returns a structured error; it is never silently ignored.

An update error can occur after arbitrary model state has already changed, so rollback is not generally possible. The simulation is therefore marked poisoned after a post-fire update failure, and later `step` or `run` calls return `SimulationPoisoned`.

## Scheduler Implementations

| Scheduler | Model surface | Cached state | Event selection |
| --- | --- | --- | --- |
| `DirectScheduler` | `EvolvingModel` | Hash map, reusable slots, one segment tree | One total-weight draw |
| `FamilyDirect` | Static families | One segment tree per family | Family total, then local weight |
| `FamilyNrm` | Static families | `Vec` of indexed family clock heaps | Minimum of family minima |
| `StaticFamilyNrm` | Static families | Recursive static clock list | Statically dispatched family minima |

`FamilyDirect`, `FamilyNrm`, and `StaticFamilyNrm` are algorithm types installed into the same generic `FamilySimulation` lifecycle. The shared engine owns `fire`, event refresh, time, termination, and poisoning; each algorithm owns only selection and cache maintenance. All calls remain monomorphized and no trait objects are used on the event path.

`FamilyNrm` is the readable generic NRM implementation. `StaticFamilyNrm` keeps clock storage and family traversal in the recursive family type. Its const generic is an inline update capacity, not a correctness limit: larger fanout spills into a reusable `Vec` instead of panicking.

## Complexity

For `F` families, `U` dependency updates, and `M_f` channels in an affected family, steady-state work is approximately:

- `FamilyDirect`: `O(F + log M_f + U log M_f)`.
- `FamilyNrm`: `O(F + U log M_f)`.
- Channel growth: the same update cost plus initialization of appended channels.

The intended workload has a small fixed `F`, so scanning family totals or minima is deliberate. Pairwise models still need a model-owned dependency index; the framework does not hide an all-species scan behind a generic fallback.

## NRM Clock Semantics

Each NRM channel stores its current propensity and absolute scheduled firing time. A fired or reinitialized channel draws a fresh exponential clock. An unaffected channel keeps its old clock. When a non-fired channel changes from `old_propensity` to `new_propensity`, the scheduler applies the Gibson-Bruck residual update:

```text
new_time = now + old_propensity / new_propensity * (old_time - now)
```

A zero propensity removes the channel from its family heap and leaves an inactive clock. If an ordinary recompute later makes it positive, it receives a fresh clock because no finite residual threshold remains.

## Randomness

`SsaRngs<C, S, E>` owns independent clock, channel-selection, and model-event streams. Direct methods consume clock and selection streams; NRM consumes clock draws for channel thresholds; mutation and other model-specific randomness consume only the event stream. Benchmarks use the same stream split in framework and handwritten implementations so scheduler comparisons are not confounded by different mutation sequences.

## Tests

Unit tests cover propensity validation, update buffers and spill behavior, weighted-tree resizing, indexed-heap behavior, and NRM residual clocks. [`tests/basic_schedulers.rs`](../tests/basic_schedulers.rs) covers the public dynamic and static APIs, arbitrary family counts beyond twelve, growth and shrink, invalid-update poisoning, slot reuse, and RNG isolation. [`tests/cache_oracle.rs`](../tests/cache_oracle.rs) compares cached Direct execution step by step with a full-rebuild reference. [`tests/exactness.rs`](../tests/exactness.rs) validates exponential waiting-time means and propensity-proportional channel frequencies for all three family schedulers with deterministic seeds.

## Benchmarks

[`benches/workloads.rs`](../benches/workloads.rs) compares framework schedulers with handwritten workload baselines for one-species birth/death, many-species birth/death, mutation-created species, and trait mutation with independently sampled child rates. Benchmark models are intentionally local to the benchmark and do not leak workload-specific types into the library API.

## Current Limits

The current clock contract covers piecewise-constant propensities. Arbitrary interior deletion and compaction remain model-owned through dense slot maps and `reinitialize`; the framework does not yet provide a species registry or generational slot allocator. Nonhomogeneous hazards, pairwise dependency indices, observers, checkpointing, and approximate methods such as tau-leaping are separate extensions.
