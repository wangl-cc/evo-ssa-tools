# Introduction

Workload, core invariants, caller contract, and terminology. Everything else in this design set assumes the definitions here.

## Purpose

`segment-cache-store` is a persistent backend for computation caches with fixed-width ordered keys and opaque serialized values. It is not a general-purpose database.

The target workload is demand-driven cache filling: callers query a sorted batch of structured inputs, compute cache misses, and publish the newly computed results for later reuse. Values are allowed to be forgotten on crash or corruption because they are recomputable, but the store must never return silently corrupted data as a valid hit.

The implementation is optimized for:

- fixed-width byte keys whose lexicographic order is the caller's desired query order
- ordered batch lookup as the primary read path
- full ordered iteration for export, migration, and analysis
- immutable segment files as the file-level sync unit
- batch publication instead of individual durable writes
- corruption-as-miss semantics

The implementation intentionally does not provide:

- transactions
- deletes or tombstones
- update-in-place
- concurrent writers
- WAL recovery
- background compaction of cold ranges
- query features beyond point lookup, ordered batch lookup, and ordered range iteration

## Scope

This crate is experimental. The on-disk format may change without migration support; rebuilding the cache from recomputable source data is the migration strategy.

The store is an insert-only logical map backed by a non-overlapping main tier and a bounded temporary patch tier. Patch is a write-amortization mechanism, not another durable level or an update history.

## Core Invariants

- One store root represents one cache namespace with one caller metadata value.
- `STORE` owns only durable namespace identity (caller metadata) and is written once during creation.
- `MANIFEST` owns the logical schema (key length, value layout) and is the only source of visible data.
- Segment files are immutable after publication.
- Each segment header is self-describing: it declares the logical schema and physical encoding needed to decode that segment. The reader trusts the header after CRC validation.
- Segment open validates that the header's logical schema (key_len, value_layout) matches the manifest's schema. The header's block checksum id must be the one fixed store checksum; unknown ids are unsupported and the segment is treated as absent. Value-payload compression is not compared — the reader uses whatever the header declares.
- Segment files not referenced by `MANIFEST` are invisible.
- Every visible logical key has exactly one value.
- Main-tier segment ranges are sorted and globally non-overlapping.
- A patch segment may overlap main ranges, but one connected normalization component contains at most one patch segment.
- Patch records never duplicate keys already visible in main or another patch.
- Missing, malformed, or corrupted data is treated as absent, never as valid output.
- Writes are serialized by one logical writer.
- Every block carries the same fixed block checksum (unseeded XXH3 128-bit); it is a store-wide constant, not a per-store or per-segment choice. Value-payload compression is a write-time parameter, not persistent store identity; each segment's header declares its own.

## Caller Contract

The store validates key length, fixed value length, manifest structure, segment structure, and block checksums; every block in every store carries the same fixed block checksum. The store cannot validate the caller's logical encoding and cache semantics.

- **Stable key encoding.** The caller must encode each logical input into the same fixed-width key bytes wherever the cache is used. The store compares keys lexicographically and does not understand their internal structure.
- **Stable value for one key.** Recommitting the exact same key/value bytes is an idempotent no-op. Reusing one key for different value bytes is a contract violation and fails before publication. Stochastic computations must therefore include their seed or replicate identity in the key.
- **Single logical writer.** Only one writer may mutate a root at a time. The implementation enforces this for cooperating processes with an advisory lock.
- **Immutable files while open.** Segment files must not be externally modified while a `Store` handle is open. The runtime may skip repeated checksum work for block parts that were already verified in the same process.

## Terminology

### Logical Data

- `Key`: fixed-width byte string used for ordering and lookup.
- `Value`: opaque serialized byte string associated with one key.
- `Entry`: caller-provided key/value pair before it is encoded into a segment.
- `Record`: key/value pair decoded from visible storage and returned by reads or scans.
- `Conflict`: two records with the same key and different value bytes. A conflict is rejected rather than resolved by arrival order or an arbitrary byte-order winner.

`Entry` means write input. `Record` means read output. Other entry-like structures should be qualified, such as `Manifest entry` or `Block index entry`.

### Tiers

- `Main tier (L1)`: sorted, globally non-overlapping segment set.
- `Patch tier (L0)`: bounded temporary overlay for one small interleaving batch. A patch may overlap main ranges but does not contain keys already visible elsewhere.
- `Normalization component`: the interval-connected set obtained by starting from a patch or incoming batch and repeatedly including every main or patch segment whose range intersects the current bounds.
- `Normalization`: replacing commit that streams one or more normalization components into new main-tier segments while retaining every unaffected main segment unchanged.
- `Cross-store merge`: strict streaming union of source and destination snapshots that rewrites only the destination components intersecting the source's key ranges and publishes one replacement snapshot.

### Catalog

- `Store root`: directory containing one cache namespace.
- `Metadata`: caller-defined compatibility bytes persisted in `STORE`.
- `STORE`: stable namespace identity descriptor containing only caller metadata.
- `MANIFEST`: atomic visible-data snapshot that also owns the logical schema (key length, value layout).
- `Manifest entry`: one visible segment reference plus tier and key range.
- `StoreSchema`: the logical data contract (key_len, value_layout) owned by the manifest and redundantly declared in each segment header.
- `SegmentEncoding`: the physical encoding choices declared by each segment header. The block checksum is a fixed store-wide constant; only value-payload compression is a write-time parameter, not persistent store identity.

### Physical Layout

- `Segment`: immutable file containing sorted records.
- `Segment header`: fixed-size self-describing prefix that declares the logical schema and physical encoding needed to decode the segment.
- `Segment footer`: variable-size suffix that is the segment completion marker and owns the sparse block index.
- `Block`: physical read unit inside a segment.
- `Block index`: segment-footer table with one entry per block.
- `KeySection`: block-local key metadata. It follows the block's `record_count:u32`, starts with `extra_prefix_len:u32`, then stores the additional prefix after the segment-owned prefix and the fixed-width suffix table.
- `ValueIndexSection`: variable-value offset table with `record_count + 1` offsets. Fixed-value blocks do not have this section.
- `ValuePayloadFrame`: raw or compressed packed user value bytes for one block.
- `LookupMetadata`: the bytes needed to prove key membership and locate values in a block. It is `KeySection` plus `ValueIndexSection` for variable-value blocks, and just `KeySection` for fixed-value blocks.
