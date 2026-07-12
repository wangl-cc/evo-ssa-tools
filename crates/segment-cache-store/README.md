# segment-cache-store

`segment-cache-store` is a standalone persistent storage backend for computation-cache workloads with fixed-width ordered keys and opaque byte values.

It is optimized for:

- append-only inserts
- merging compatible worker stores
- ordered batch lookup
- full ordered iteration
- corruption-as-miss semantics

This crate is intentionally narrower than a general-purpose database. It does not provide transactions, deletes, WAL recovery, or background compaction of cold ranges. The foreground maintenance operations are explicit: normalization folds live patch segments into the main tier, and garbage collection removes unreferenced segment files. Values for one key are expected to be deterministic and semantically immutable: if duplicate visible copies exist, reads, normalization, and store merges keep the lexicographically smallest value bytes rather than applying last-writer-wins updates.

The crate currently supports Unix targets only. Its positioned reads, directory durability, and advisory writer lock semantics rely on Unix filesystem APIs.

## Basic Use

```rust,no_run
use segment_cache_store::{
    BlockChecksumKind, CreateOptions, OpenOptions, Store, StoreMetadata,
};

let metadata = StoreMetadata::from_text("my-cache-schema-v1");
let create_options = CreateOptions::new(16, metadata.clone(), BlockChecksumKind::None)?;
let store = Store::create("cache-root", create_options)?;

let mut batch = segment_cache_store::WriteBatch::new();
batch.push(&[0; 16], b"serialized value");
store.commit_batch(batch)?;

let reopened = Store::open("cache-root", OpenOptions::read_write(metadata))?;
let value = reopened.fetch_one(&[0; 16])?;
# Ok::<_, segment_cache_store::Error>(())
```

One store root has one fixed key length, one value layout, one block checksum implementation, one value-payload compression kind, and one caller-defined metadata namespace. Callers select the checksum explicitly when constructing `CreateOptions`. Value-payload compression defaults to `ValuePayloadCompressionKind::None`; optional feature flags expose compression-capable kinds such as `ValuePayloadCompressionKind::Lz4` and `ValuePayloadCompressionKind::ZstdLevel1` for `CreateOptions::with_value_payload_compression(kind)`. Per-commit `ValuePayloadCompressionPolicy` thresholds control when newly written compression-capable blocks actually keep compressed frames. Published segment files are immutable. The main tier remains globally non-overlapping; a bounded patch tier can temporarily overlap main segments so small interleaving commits avoid immediate rebuild. When the patch tier reaches its configured bound, the store normalizes the touched range and atomically publishes a replacement `MANIFEST`. Callers can also run `Store::normalize()` explicitly before a read-heavy phase, `Store::merge_from(&source)` to atomically import visible records from a compatible worker store, and `Store::garbage_collect()` explicitly when retired segment files should be reclaimed.

## Feature Flags

The default feature set enables `checksum-rapidhash`, which exposes `BlockChecksumKind::RapidHashV3_64`. `BlockChecksumKind::None` is always available; it stores no per-block checksum bytes, but catalog metadata and manifest-to-segment identity checks still use internal integrity checks when a store is opened. `checksum-crc32c` exposes `BlockChecksumKind::Crc32c` as an optional block checksum implementation; CRC32C is still used internally for fixed catalog and segment structural checks. The concrete checksum features transitively enable the `block-checksum` capability marker. `value-compression-lz4` and `value-compression-zstd` expose optional block-level value-payload compression algorithms and transitively enable the `value-compression` capability marker. Enabling either marker alone adds no concrete algorithm. `CreateOptions::new(key_len, metadata, kind)` has the same signature for every feature combination.

Internal design and evaluation notes live in:

- [docs/design.md](docs/design.md)
- [docs/benchmark.md](docs/benchmark.md)

Useful commands:

```bash
cargo run -p scs -- stats cache-root
cargo run -p scs -- get cache-root <hex-key>
cargo run -p scs -- merge cache-root worker-cache-root
cargo run -p scs -- compact cache-root
cargo bench -p segment-cache-store --bench comparison
cargo bench -p segment-cache-store --bench ordered_lookup
cargo bench -p segment-cache-store --bench append_publish
cargo bench -p segment-cache-store --bench parameter_evolution
cargo bench -p segment-cache-store --features value-compression-lz4,value-compression-zstd --bench compression
cargo run -p segment-cache-store --example space_usage --offline
```
