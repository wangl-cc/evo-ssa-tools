# segment-cache-store

`segment-cache-store` is a standalone persistent storage backend for computation-cache workloads with fixed-width ordered keys and opaque byte values.

It is optimized for:

- append-only inserts
- ordered batch lookup
- full ordered iteration
- corruption-as-miss semantics

This crate is intentionally narrower than a general-purpose database. It does not provide transactions, deletes, compaction, or WAL recovery.

Internal design and evaluation notes live in:

- [docs/design.md](docs/design.md)
- [docs/benchmark.md](docs/benchmark.md)

Useful commands:

```bash
cargo bench -p segment-cache-store --bench workload
cargo run -p segment-cache-store --example space_usage --offline
```
