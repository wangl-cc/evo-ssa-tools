# ssa-workflow-next (prototype)

Shape check for the caller-driven `ssa-workflow` redesign. Not for production: in-memory store only, no deduplication, no per-item errors, no interrupts, no progress, no flush policy.

What it demonstrates (see `tests/ecdna_shape.rs`):

- `Kernel` with associated types: callers never annotate types; random streams are requested lazily through `Streams`, so there is no deterministic/stochastic split.
- `Stage::root` / `Stage::after` bind a kernel to a store and a parent path once per run.
- `stage.request(&upstream, param)` builds a `Req` whose keys are `canonical(param) ++ upstream key`. The parent is erased as `&dyn Upstream<U>`, so types do not nest.
- `stage.request_fan(&upstream, &params)` fans out: every upstream item under every parameter, parameter-major. Keys match one `request` per parameter, and each upstream item is pulled at most once.
- `req.materialize()` does one ordered lookup, loads hits, asks upstream only for misses, computes misses with the request's `Exec` workers, and writes them. A chain therefore runs stage by stage within a block, and an all-hit rerun never touches upstream.
- A materialized `Block` can be lent to downstream requests (staged outputs, shared branches).
- `Exec::threads(n)` runs a stage on `n` scoped worker threads of our own (no Rayon). Workers claim items from a shared counter, so at most `n` items are in flight, and there is no work stealing to nest into a kernel's own Rayon pool.
- Worker state (for example an assembled operator) is checked out once per worker thread from a per-stage pool and survives across blocks.
