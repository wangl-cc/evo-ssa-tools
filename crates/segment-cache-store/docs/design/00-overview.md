# segment-cache-store Design Overview

`segment-cache-store` is a persistent backend for computation caches with fixed-width ordered keys and opaque serialized values. It is not a general-purpose database.

## Documents

Agreed local target:

- [01 Introduction](01-introduction.md) — workload, core invariants, caller contract, terminology. Read this first.
- [02 Architecture](02-architecture.md) — module map and process model.
- [03 Format](03-format.md) — on-disk bytes: directory layout, catalog, segment, block, format limits.
- [04 Operations](04-operations.md) — open, read, write, and cross-store merge algorithms, with the runtime optimizations of each path.
- [05 Corruption](05-corruption.md) — corruption-as-miss semantics and the trust boundary.

Agreed remote extension (not implemented):

- [06 Remote](06-remote/00-design.md) — remote repository design, with its [version format](06-remote/01-version-format.md) and [implementation plan](06-remote/02-implementation-plan.md).

Related documents outside this set:

- [Benchmarking](../benchmark.md) — benchmark suite and workload definitions; results live in [benchmark-baselines/](../../benchmark-baselines/README.md).

## Conventions

- Documents 01–05 define the agreed local target contract. A behavior that the current implementation has not reached is marked inline as a "current implementation note" at the exact point where reading the target as current behavior would be misleading.
- [README.md](../../README.md) and public Rust API documentation describe the current implementation. They must not claim an unimplemented target behavior.
- [06-remote/](06-remote/) defines the future remote repository extension. It is intentionally separate from local v1 bytes and every remote document is marked as unimplemented.
- The complete implementation delta, validation work, phases, and unresolved numeric policy choices live in [TODO.md](../../TODO.md). A decided invariant belongs in these design documents rather than remaining an open TODO.
