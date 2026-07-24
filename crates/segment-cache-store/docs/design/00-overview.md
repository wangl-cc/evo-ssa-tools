# segment-cache-store Design Overview

`segment-cache-store` is a persistent backend for computation caches with fixed-width ordered keys and opaque serialized values. It is not a general-purpose database.

## Documents

Current design:

- [01 Introduction](01-introduction.md) — workload, core invariants, caller contract, terminology. Read this first.
- [02 Architecture](02-architecture.md) — module map and process model.
- [03 Format](03-format.md) — on-disk bytes: directory layout, catalog, segment, block, format limits.
- [04 Operations](04-operations.md) — open, read, write, and cross-store merge algorithms, with the runtime optimizations of each path.
- [05 Corruption](05-corruption.md) — corruption-as-miss semantics and the trust boundary.

Agreed future direction (not implemented):

- [06 Remote](06-remote/00-design.md) — remote repository design, with its [version format](06-remote/01-version-format.md) and [implementation plan](06-remote/02-implementation-plan.md).

Related documents outside this set:

- [Benchmarking](../benchmark.md) — benchmark suite and workload definitions; results live in [benchmark-baselines/](../../benchmark-baselines/README.md).

## Conventions

- The numbered documents describe the agreed design of the current store. Where the v1 implementation has not caught up with it, the delta is marked inline as a "v1 implementation note" at the exact spot an implementer would be misled, and the full delta list is tracked in [TODO.md](../../TODO.md).
- [06-remote/](06-remote/) is the one exception: it holds the agreed future direction for the remote repository, each document marked as not implemented. When that work lands, its content promotes into the current-design documents in the same pull request as the code.
- Design changes land in the same pull request as their implementation (or immediately ahead of it). Agreed-but-unimplemented design does not accumulate in the numbered documents ahead of code.
- Status, phases, task lists, and open decisions live in [TODO.md](../../TODO.md) and pull requests, not in these documents.
