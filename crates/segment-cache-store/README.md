# segment-cache-store

`segment-cache-store` is a standalone persistent storage backend for computation-cache workloads with fixed-width ordered keys and opaque byte values.

It is optimized for:

- append-only inserts
- merging compatible worker stores
- ordered batch lookup
- full ordered iteration
- corruption-as-miss semantics

This crate is intentionally narrower than a general-purpose database. It does not provide transactions, deletes, WAL recovery, or background compaction of cold ranges. The foreground maintenance operations are explicit: normalization folds bounded temporary patch data into the main tier, and garbage collection removes unreferenced segment files. The logical store contains at most one value for each key. Recommitting identical key/value bytes is an idempotent no-op; reusing a key for different value bytes fails as a conflict instead of applying last-writer-wins or an arbitrary byte-order winner.

The crate currently supports Unix targets only. Its positioned reads, directory durability, and advisory writer lock semantics rely on Unix filesystem APIs.

## Design Status

This README and the public Rust API documentation describe the current implementation. The numbered [design documents](docs/design/00-overview.md) define the agreed pre-release target, and [TODO.md](TODO.md) records the implementation delta. In particular, the current code still permits one Patch per interval-connected component, reports `KeyConflict` for differing bytes under one key, and uses strict cursors in several ordinary write and merge paths; the target replaces those behaviors with Main-owned PatchSets, destination-biased first-healthy-winner semantics, and corruption-as-absence across ordinary cache operations.

## Basic Use

```rust,no_run
use segment_cache_store::{CreateOptions, OpenOptions, Store, StoreMetadata};

let metadata = StoreMetadata::from_text("my-cache-schema-v1");
let create_options = CreateOptions::new(16, metadata.clone())?;
let store = Store::create("cache-root", create_options)?;

let mut batch = segment_cache_store::WriteBatch::new();
batch.push(&[0; 16], b"serialized value");
store.commit_batch(batch)?;

let reopened = Store::open("cache-root", OpenOptions::read_write(metadata))?;
let value = reopened.fetch_one(&[0; 16])?;
# Ok::<_, segment_cache_store::Error>(())
```

One store root has one fixed key length, one value layout, one value-payload compression kind, and one caller-defined metadata namespace. Local format version 1 fixes canonical XXH3-128 checksums for MANIFEST, segment footer, and block sections; the algorithm is not STORE metadata or a runtime choice. Value-payload compression defaults to `ValuePayloadCompressionKind::None`; optional feature flags expose compression-capable kinds such as `ValuePayloadCompressionKind::Lz4` and `ValuePayloadCompressionKind::ZstdLevel1` for `CreateOptions::with_value_payload_compression(kind)`. Per-commit `ValuePayloadCompressionPolicy` thresholds control when newly written compression-capable blocks actually keep compressed frames. Published segment files are immutable. The main tier remains globally non-overlapping; each interval-connected region may temporarily hold one small patch segment so one interleaving commit avoids an immediate rebuild. A second overlapping commit normalizes only that region and atomically publishes a replacement `MANIFEST`. Callers can also run `Store::normalize()` explicitly before a read-heavy phase, `Store::merge_from(&source)` to atomically merge a compatible worker store while rewriting only the intersecting key ranges, and `Store::garbage_collect()` explicitly when retired segment files should be reclaimed.

## Feature Flags

`value-compression-lz4` and `value-compression-zstd` expose optional block-level value-payload compression algorithms and transitively enable the `value-compression` capability marker. Enabling the marker alone adds no concrete algorithm. Checksums are always compiled in: MANIFEST, segment footer, and every block section carry canonical XXH3-128 checksums, while CRC32C protects only the short fixed segment header; neither algorithm is feature-gated.

Internal design and evaluation notes live in:

- [docs/design/](docs/design/00-overview.md) — agreed local target (00–05: introduction, architecture, format, operations, corruption semantics) plus the future remote extension (06-remote/, not implemented)
- [docs/benchmark.md](docs/benchmark.md) — benchmark suite and workload definitions; results live in [benchmark-baselines/](benchmark-baselines/README.md)

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
