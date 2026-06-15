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

One store root has one fixed key length, one value layout, one block checksum implementation, and one caller-defined metadata namespace. The default block checksum is `BlockChecksumKind::RapidHashV3_64`; callers can select another built-in implementation through `CreateOptions::with_block_checksum(kind)`. Published segment files are immutable. The main tier remains globally non-overlapping; a bounded patch tier can temporarily overlap main segments so small interleaving commits avoid immediate rebuild. When the patch tier reaches its configured bound, the store normalizes the touched range and atomically publishes a replacement `MANIFEST`. Callers can also run `Store::normalize()` explicitly before a read-heavy phase.

## Feature Flags

The default feature set enables `checksum-rapidhash`, which exposes `BlockChecksumKind::RapidHashV3_64` and makes `CreateOptions::new` use it as the default block checksum. `BlockChecksumKind::None` is always available. `checksum-crc32c` exposes `BlockChecksumKind::Crc32c` as an optional block checksum implementation; CRC32C is still used internally for fixed catalog and segment structural checks. If default features are disabled, use `CreateOptions::new_with_block_checksum(key_len, metadata, kind)` so the checksum choice remains explicit.

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
