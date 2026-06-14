# Benchmarking

## Purpose

The benchmark suite evaluates `segment-cache-store` as a storage backend for computation caches. It models storage-level behavior rather than scientific model semantics: fixed-width ordered keys, narrow per-namespace value-size profiles, ordered batch reads, append-style publish, parameter-space evolution, and full export scans.

One benchmark target is used for horizontal comparison against other embedded stores. The remaining targets are segment-only diagnostics, so internal design questions do not multiply every scenario by every baseline backend.

## Benchmark Targets

- `comparison`: cross-backend summary benchmark. It compares `segment-cache-store`, tuned `fjall3`, and `redb` on ordered fetch, sparse ordered fetch, full iteration, append publish, and the small-profile parameter-evolution workload.
- `ordered_lookup`: segment-only read-path benchmark with checksum verification enabled. It measures dense ordered lookup, sparse ordered lookup, large-value borrowed-vs-owned fetch APIs, large-value block-size sensitivity, L0 overlay ordered lookup, and L0 overlay scan amplification.
- `append_publish`: segment-only write-path benchmark with checksum verification enabled. It measures sorted and unsorted batch publish into fresh stores.
- `parameter_evolution`: segment-only cache-evolution benchmark with checksum verification enabled. It measures rebuild-vs-L0 behavior for middle inserts and repeated axis changes.

`segment_no_crc` appears only in `comparison`. It opens the segment store read-only with block checksum verification disabled, so it is an explicit upper-bound variant for comparing against engines that do not validate user value bytes on read. All segment-only diagnostics keep checksum verification enabled because they are meant to measure the cache-safe implementation.

## Dataset Shape

The benchmark uses `128B` fixed-width keys. The synthetic key layout has long stable prefixes, ordered parameter axes, and a repetition-like suffix, which is closer to canonical-encoded parameter sweeps than random short keys.

Value profiles are narrow within one namespace:

- `small = 64B ± 8B`
- `small_fixed = 64B`
- `medium = 1KB ± 128B`
- `large = 16KB ± 2KB`

`small_fixed` uses `CreateOptions::with_fixed_value_len`, so the segment backend exercises the fixed-value block layout automatically for that profile. The benchmark intentionally does not mix small, medium, and large values inside one namespace profile.

Each profile currently uses `16,384` records for the standard dataset. The parameter-evolution datasets are separate structured grids that repeatedly change active `x` and `y` axes.

## Baselines

`fjall3` uses a workload-shaped, no-compression configuration derived from the surrounding project: large cache, higher cached-file limit, point-read-hit expectation, profile-sensitive data block tuning, and key-value separation for large values. Compression is omitted because benchmark values are synthetic bytes and the goal is to measure storage layout rather than compression ratio.

`redb` is included as a mature embedded copy-on-write B-tree baseline. It uses one raw byte table and no domain-specific adapter layer.

## Workloads

### `comparison_ordered_fetch`

Fetches the full ordered stream of known keys and touches every returned value. This is the primary hot-path comparison. Variants: `segment`, `segment_no_crc`, `fjall3`, and `redb`.

### `comparison_sparse_ordered_fetch`

Fetches an ordered 1/16 subset of known keys. This tests ordered but sparse reuse, where larger physical blocks can increase read amplification. Variants: `segment`, `segment_no_crc`, `fjall3`, and `redb`.

### `comparison_iter_all`

Scans every visible record in order and touches every value. This represents export, migration, and full ordered readback. Variants: `segment`, `segment_no_crc`, `fjall3`, and `redb`.

### `comparison_append_publish`

Publishes one sorted batch into a fresh store. This is the cleanest cross-backend write comparison for append-style cache materialization. Variants: `segment`, `fjall3`, and `redb`.

### `comparison_axis_change_rounds`

Runs several cache-shaped rounds where the active Cartesian-product parameter set changes over time. Each round does ordered lookup, touches hit values, and commits only misses. This target is registered only for the `small` profile to keep the cross-backend matrix bounded.

### `ordered_fetch`

Segment-only dense ordered lookup with checksum verification enabled. This is a stable internal regression target for the default read path.

### `sparse_ordered_fetch`

Segment-only sparse ordered lookup with checksum verification enabled. It sweeps block sizes `16K`, `32K`, `64K`, and `256K` to expose the sparse-read amplification tradeoff.

### `large_value_block_size`

Segment-only dense ordered lookup for the `large` profile with block sizes `16K`, `64K`, `256K`, and `512K`. This answers whether large-value performance is blocked by block sizing or by value movement and checksum cost.

### `large_value_fetch_api`

Segment-only dense ordered lookup for the `large` profile comparing the borrowed visitor path with the owned `fetch_many_ordered` path at `16K` and `512K` block sizes. This isolates API-level value materialization cost from physical block-size effects.

### `overlay_ordered_fetch`

Segment-only L0 overlay read benchmark. `main_only` stores every record in the main tier; `patch_1`, `patch_2`, `patch_4`, and `patch_8` store one eighth of the records in patch segments and verify that all stores expose the same logical records.

### `overlay_iter_all`

Segment-only L0 overlay scan benchmark. It uses the same logical layout as `overlay_ordered_fetch`, but scans every visible record through `visit_all`, exercising the merge cursor used when patch segments are present.

### `append_publish`

Segment-only publish benchmark for `sorted_batch` and `unsorted_batch`. The sorted variant models callers that already emit canonical key order; the unsorted variant includes backend sorting cost.

### `middle_insert_then_read`

Segment-only parameter-space insertion benchmark. `rebuild_new_store_then_read` rebuilds a new normalized store, while `l0_chunked_insert_then_read` publishes the inserted middle range in chunks and then reads the expanded key set through the L0 overlay.

### `axis_change_rounds`

Segment-only repeated parameter-evolution benchmark. It prints the dry-run rewrite amplification (`merged_records / inserted`) so L0 normalization behavior can be tracked without comparing against unrelated database internals.

## Commands

Run the horizontal comparison suite:

```bash
cargo bench -p segment-cache-store --bench comparison
```

Run segment-only read-path diagnostics:

```bash
cargo bench -p segment-cache-store --bench ordered_lookup
```

Run segment-only publish diagnostics:

```bash
cargo bench -p segment-cache-store --bench append_publish
```

Run segment-only parameter-evolution diagnostics:

```bash
cargo bench -p segment-cache-store --bench parameter_evolution
```

Run one Criterion subgroup:

```bash
cargo bench -p segment-cache-store --bench comparison -- comparison_ordered_fetch
```

Measure space amplification:

```bash
cargo run -p segment-cache-store --example space_usage --offline
```

## Space Metric

Space amplification is treated as a first-class metric. The measurement script lives in [examples/space_usage.rs](../examples/space_usage.rs) and reports logical bytes, directory size, and amplification factor for `segment-cache-store`, `fjall3`, and `redb`.

## Historical Snapshot

The numbers below are local historical smoke-run results from the prototype benchmark suite before the target rename. They are useful for rough shape and regression awareness, not as publication-grade measurements.

### Ordered fetch

| namespace | segment | fjall3 | redb |
| --- | ---: | ---: | ---: |
| small | 1.24-1.27 ms | 6.25-6.31 ms | 3.89-4.27 ms |
| small_fixed | 1.04-1.05 ms | 6.22-6.46 ms | 3.47-3.48 ms |
| medium | 4.87-5.11 ms | 6.54-6.58 ms | 5.14-5.18 ms |
| large | 61.89-63.39 ms | 26.65-27.14 ms | 22.64-25.75 ms |

### Append publish

| namespace | segment sorted | fjall3 | redb |
| --- | ---: | ---: | ---: |
| small | 64.29-66.92 ms | 119.88-124.82 ms | 73.79-75.89 ms |
| small_fixed | 64.63-67.93 ms | 120.10-125.13 ms | 75.41-79.90 ms |
| medium | 89.38-91.70 ms | 136.55-142.92 ms | 117.88-123.65 ms |
| large | 819.94-909.09 ms | 1.18-1.29 s | 752.32-816.80 ms |

### Space amplification

| namespace | logical bytes | segment | fixed layout | fjall3 | redb |
| --- | ---: | ---: | ---: | ---: | ---: |
| small | 3,145,563 | 1.077x | n/a | 21.336x | 2.679x |
| small_fixed | 3,145,728 | 1.077x | 1.035x | 21.334x | 2.678x |
| medium | 18,870,633 | 1.052x | n/a | 3.556x | 1.785x |
| large | 270,504,438 | 1.037x | n/a | 1.253x | 1.992x |

## Reading Results

The benchmark should be read as a shape test: whether the backend stays in the right performance class, whether it wins on the workload it is designed for, and whether it preserves its space-efficiency advantage. Winning every microbenchmark is less important than keeping ordered cache replay, append publish, corruption-safe reads, and file-level stability coherent in one backend.
