# Benchmarking

## Purpose

The benchmark suite evaluates `segment-cache-store` as a storage backend for computation caches. It models storage-level behavior rather than scientific model semantics: fixed-width ordered keys, narrow per-namespace value-size profiles, ordered batch reads, append-style publish, parameter-space evolution, and full export scans.

One benchmark target is used for horizontal comparison against other embedded stores. The remaining targets are segment-only diagnostics, so internal design questions do not multiply every scenario by every baseline backend.

## Benchmark Targets

- `comparison`: cross-backend summary benchmark. It compares `segment-cache-store`, tuned `fjall3`, and `redb` on ordered fetch, clustered sparse ordered fetch, full iteration, append publish, and the small-profile parameter-evolution workload.
- `ordered_lookup`: segment-only read-path benchmark with checksum verification enabled. It measures dense ordered lookup, clustered/random sparse ordered lookup, large-value borrowed-vs-owned fetch APIs, large-value block-size sensitivity, L0 overlay ordered lookup, and L0 overlay scan amplification.
- `append_publish`: segment-only write-path benchmark with checksum verification enabled. It measures sorted and unsorted batch publish into fresh stores.
- `parameter_evolution`: segment-only cache-evolution benchmark with checksum verification enabled. It measures rebuild-vs-L0 behavior for middle inserts and repeated axis changes.
- `compression`: segment-only value-payload compression benchmark, available with `--features value-compression-lz4,value-compression-zstd`. It compares uncompressed stores against LZ4-created and Zstandard-level-1-created stores using the default writer-side compression policy, reports store bytes, and measures ordered fetch, full iteration, and append publish.

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

### `comparison_clustered_sparse_ordered_fetch`

Fetches a clustered ordered 1/16 subset of known keys: each 256-record window contributes one contiguous 16-record burst. This keeps the sparse ratio comparable to the old evenly-spaced stress test while better modeling bursty parameter subsets where some blocks have multiple hits and others have none. Variants: `segment`, `segment_no_crc`, `fjall3`, and `redb`.

### `comparison_iter_all`

Scans every visible record in order and touches every value. This represents export, migration, and full ordered readback. Variants: `segment`, `segment_no_crc`, `fjall3`, and `redb`.

### `comparison_append_publish`

Publishes one sorted batch into a fresh store. This is the cleanest cross-backend write comparison for append-style cache materialization. Variants: `segment`, `fjall3`, and `redb`.

### `comparison_axis_change_rounds`

Runs several cache-shaped rounds where the active Cartesian-product parameter set changes over time. Each round does ordered lookup, touches hit values, and commits only misses. This target is registered only for the `small` profile to keep the cross-backend matrix bounded.

### `ordered_fetch`

Segment-only dense ordered lookup with checksum verification enabled. This is a stable internal regression target for the default read path.

### `clustered_sparse_ordered_fetch`

Segment-only clustered sparse ordered lookup with checksum verification enabled. It sweeps block sizes `16K`, `32K`, `64K`, and `256K` to expose sparse-read amplification when hits have local bursts.

### `random_sparse_ordered_fetch`

Segment-only seeded random sparse ordered lookup with checksum verification enabled. It keeps the query stream sorted after sampling, but the selected keys are not evenly spaced. This tests a less regular sparse shape than the old every-16th-key stress case.

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

### `compression_ordered_fetch`

Segment-only dense ordered lookup for the `large` profile with checksum verification enabled. It uses `random_bytes`, `template_noise`, `correlated_series`, and `repeated_runs` value streams, then compares uncompressed stores against LZ4-created and Zstandard-level-1-created stores with the default `ValuePayloadCompressionPolicy`. `template_noise` mixes repeated 32-byte templates with random 32-byte tails to model serialized bytes that are moderately compressible but not run-length-like. `correlated_series` stores a quantized mean-reverting random walk as little-endian integers, modeling trajectory-like numeric arrays that are random but locally structured. This exposes read-side decompression overhead and whether medium-compressibility payloads are worth compressing.

### `compression_iter_all`

Segment-only full scan for the `large` profile with `random_bytes`, `template_noise`, `correlated_series`, and `repeated_runs` value streams. It uses the default `16K` logical block split target and compares uncompressed versus LZ4-created and Zstandard-level-1-created stores under the default compression policy.

### `compression_append_publish`

Segment-only publish benchmark for the `large` profile with `random_bytes`, `template_noise`, `correlated_series`, and `repeated_runs` value streams. It uses the default `16K` logical block split target and compares write-time compression cost against the reduced bytes written when payloads compress well.

### `middle_insert_then_read`

Segment-only parameter-space insertion benchmark. `rebuild_new_store_then_read` rebuilds a new normalized store, while `l0_chunked_insert_then_read` publishes the inserted middle range in chunks and then reads the expanded key set through the L0 overlay.

### `axis_change_rounds`

Segment-only repeated parameter-evolution benchmark. It prints the dry-run rewrite amplification (`output_records / inserted`) so L0 normalization behavior can be tracked without comparing against unrelated database internals.

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

### Large read baseline before K/V checksum split

Targeted local run after lazy key materialization, before K/V checksum split and before replacing the old evenly-spaced sparse stress case with clustered/random sparse workloads:

```bash
cargo bench -p segment-cache-store --bench ordered_lookup --offline -- large/ --sample-size 20 --warm-up-time 1 --measurement-time 2 --noplot
```

| workload | result |
| --- | ---: |
| `large/ordered_fetch/default_block` | 47.8 ms |
| `large/sparse_ordered_fetch/block_16k` | 3.29 ms |
| `large/sparse_ordered_fetch/block_32k` | 4.82 ms |
| `large/sparse_ordered_fetch/block_64k` | 8.25 ms |
| `large/sparse_ordered_fetch/block_256k` | 28.25 ms |
| `large/large_value_fetch_api/borrowed_block_16k` | 47.0 ms |
| `large/large_value_fetch_api/owned_block_16k` | 56.8 ms |
| `large/large_value_fetch_api/borrowed_block_512k` | 35.2 ms |
| `large/large_value_fetch_api/owned_block_512k` | 49.5 ms |
| `large/large_value_block_size/block_16k` | 47.5 ms |
| `large/large_value_block_size/block_64k` | 41.9 ms |
| `large/large_value_block_size/block_256k` | 36.2 ms |
| `large/large_value_block_size/block_512k` | 35.6 ms |

### Large sparse after K/V checksum split

Targeted local run after splitting each block checksum into lookup metadata and value-payload checksums. Dense ordered lookup still loads complete blocks; sparse ordered lookup first validates metadata and only loads value payload for blocks that contain at least one matching key.

```bash
cargo bench -p segment-cache-store --bench ordered_lookup --offline -- large/clustered_sparse_ordered_fetch --sample-size 20 --warm-up-time 1 --measurement-time 2 --noplot
cargo bench -p segment-cache-store --bench ordered_lookup --offline -- large/random_sparse_ordered_fetch --sample-size 20 --warm-up-time 1 --measurement-time 2 --noplot
cargo bench -p segment-cache-store --bench ordered_lookup --offline -- large/ordered_fetch/default_block --sample-size 20 --warm-up-time 1 --measurement-time 2 --noplot
cargo bench -p segment-cache-store --bench comparison --offline -- large/comparison_clustered_sparse_ordered_fetch --sample-size 20 --warm-up-time 1 --measurement-time 2 --noplot
```

| workload | 16K | 32K | 64K | 256K |
| --- | ---: | ---: | ---: | ---: |
| `large/clustered_sparse_ordered_fetch` | 2.84 ms | 3.52 ms | 3.16 ms | 4.69 ms |
| `large/random_sparse_ordered_fetch` | 3.23 ms | 5.82 ms | 8.93 ms | 20.85 ms |

| workload | result |
| --- | ---: |
| `large/ordered_fetch/default_block` | 47.15 ms |

| backend | `large/comparison_clustered_sparse_ordered_fetch` |
| --- | ---: |
| segment | 2.86 ms |
| segment_no_crc | 1.97 ms |
| fjall3 | 1.24 ms |
| redb | 0.85 ms |

The old evenly spaced sparse stress case made every selected key land predictably across the key stream. The current clustered workload selects 16 contiguous keys out of each 256-record window, while the random workload uses a seeded 1/16 sample and then preserves sorted lookup order. The split checksum helps most when a loaded block has no matching key, because the reader can validate metadata and skip value-payload IO. It does not avoid payload validation for blocks that contain hits.

### Large reads after process-local verification reuse

Targeted local run after adding process-local per-block verification state. Criterion warmup validates the relevant blocks first, so these numbers represent warm repeated reads in one process, not first-touch cold reads from a fresh open:

```bash
cargo bench -p segment-cache-store --bench ordered_lookup --offline -- large/ordered_fetch/default_block --sample-size 20 --warm-up-time 1 --measurement-time 2 --noplot
cargo bench -p segment-cache-store --bench ordered_lookup --offline -- large/clustered_sparse_ordered_fetch --sample-size 20 --warm-up-time 1 --measurement-time 2 --noplot
cargo bench -p segment-cache-store --bench ordered_lookup --offline -- large/random_sparse_ordered_fetch --sample-size 20 --warm-up-time 1 --measurement-time 2 --noplot
cargo bench -p segment-cache-store --bench comparison --offline -- large/comparison_clustered_sparse_ordered_fetch --sample-size 20 --warm-up-time 1 --measurement-time 2 --noplot
```

| workload | result |
| --- | ---: |
| `large/ordered_fetch/default_block` | 37.93 ms |

| workload | 16K | 32K | 64K | 256K |
| --- | ---: | ---: | ---: | ---: |
| `large/clustered_sparse_ordered_fetch` | 2.44 ms | 3.42 ms | 3.18 ms | 4.38 ms |
| `large/random_sparse_ordered_fetch` | 2.99 ms | 5.47 ms | 8.29 ms | 18.49 ms |

| backend | `large/comparison_clustered_sparse_ordered_fetch` |
| --- | ---: |
| segment | 2.32 ms |
| segment_no_crc | 3.31 ms |
| fjall3 | 1.59 ms |
| redb | 1.24 ms |

The segment-only runs are the cleaner signal: dense large ordered fetch improved by roughly 20%, clustered sparse 16K by roughly 13%, and random sparse 16K by roughly 8%. The comparison run was noisier in this local pass: `segment_no_crc`, `fjall3`, and `redb` all regressed relative to their previous local baselines, so the cross-backend row should be treated as a smoke result rather than a stable ranking.

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
