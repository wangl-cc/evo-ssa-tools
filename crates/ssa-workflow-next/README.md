# ssa-workflow-next (prototype)

Shape check for the caller-driven `ssa-workflow` redesign. Not for production: in-memory store only, no deduplication, no per-item errors, no interrupts, no progress, no flush policy.

What it demonstrates (see `tests/ecdna_shape.rs`):

- `Kernel` with associated types: callers never annotate types; RNG is requested lazily through `Cx`, so there is no deterministic/stochastic split.
- `Stage::root` / `Stage::after` bind a kernel to a store and a parent path once per run.
- `stage.request(&upstream, param)` builds a `Req` whose keys are `canonical(param) ++ upstream key`. The parent is erased as `&dyn Upstream<U>`, so types do not nest.
- `req.materialize()` does one ordered lookup, loads hits, asks upstream only for misses, computes misses on the request's `Exec` pool, and writes them. A chain therefore runs stage by stage within a block, and an all-hit rerun never touches upstream.
- A materialized `Block` can be lent to downstream requests (staged outputs, shared branches).
- Worker state (for example an assembled operator) lives in a per-stage pool and survives across blocks.
