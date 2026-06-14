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

One store root has one fixed key length, one value layout, and one caller-defined metadata namespace. Published segment files are immutable. The main tier remains globally non-overlapping; a bounded patch tier can temporarily overlap main segments so small interleaving commits avoid immediate rebuild. When the patch tier reaches its configured bound, the store normalizes the touched range and atomically publishes a replacement `MANIFEST`. Callers can also run `Store::normalize()` explicitly before a read-heavy phase.

Internal design and evaluation notes live in:

- [docs/design.md](docs/design.md)
- [docs/benchmark.md](docs/benchmark.md)

Useful commands:

```bash
cargo bench -p segment-cache-store --bench comparison
cargo bench -p segment-cache-store --bench ordered_lookup
cargo bench -p segment-cache-store --bench append_publish
cargo bench -p segment-cache-store --bench parameter_evolution
cargo run -p segment-cache-store --example space_usage --offline
```
