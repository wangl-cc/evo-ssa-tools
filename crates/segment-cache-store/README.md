# segment-cache-store

`segment-cache-store` is a standalone persistent storage backend for computation-cache workloads with fixed-width ordered keys and opaque byte values.

It is optimized for:

- append-only inserts
- ordered batch lookup
- full ordered iteration
- corruption-as-miss semantics

This crate is intentionally narrower than a general-purpose database. It does not provide transactions, deletes, compaction, or WAL recovery.

## Basic Use

```rust,no_run
use segment_cache_store::{CreateOptions, OpenOptions, Store, StoreMetadata};

let metadata = StoreMetadata::from_text("my-cache-schema-v1");
let store = Store::create("cache-root", CreateOptions::new(16, metadata.clone()))?;

let mut batch = store.begin_batch();
batch.push(&[0; 16], b"serialized value")?;
store.commit_batch(batch.mark_sorted())?;

let reopened = Store::open("cache-root", OpenOptions::new(metadata))?;
let value = reopened.fetch_one(&[0; 16])?;
# Ok::<_, segment_cache_store::Error>(())
```

One store root has one fixed key length, one value layout, and one caller-defined metadata namespace. Published segments are immutable and globally non-overlapping. New commits may append at the tail or insert into gaps, but they may not overlap an existing visible segment range.

Internal design and evaluation notes live in:

- [docs/design.md](docs/design.md)
- [docs/benchmark.md](docs/benchmark.md)

Useful commands:

```bash
cargo bench -p segment-cache-store --bench workload
cargo run -p segment-cache-store --example space_usage --offline
```
