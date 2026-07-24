# Benchmark Suite

The benchmark workloads: what is measured, on what data, against which backends, and how to run it. Measurement results live in [benchmark-baselines/](../benchmark-baselines/README.md); this document intentionally contains no numbers.

## Purpose

The benchmark suite evaluates `segment-cache-store` as a storage backend for computation caches. It models storage-level behavior rather than scientific model semantics: fixed-width ordered keys, narrow per-namespace value-size profiles, ordered batch reads, append-style publish, parameter-space evolution, and full export scans.

One benchmark target is used for horizontal comparison against other embedded stores. The remaining targets are segment-only diagnostics, so internal design questions do not multiply every scenario by every baseline backend.

## Benchmark Targets

- `comparison`: cross-backend summary benchmark. It compares `segment-cache-store`, tuned `fjall3`, and `redb` on ordered fetch, clustered sparse ordered fetch, full iteration, append publish, and the small-profile parameter-evolution workload.
- `ordered_lookup`: segment-only read-path benchmark with checksum verification enabled. It measures dense ordered lookup, clustered/random sparse ordered lookup, cold process-local checksum verification, large-value borrowed-vs-owned fetch APIs, large-value block-size sensitivity, L0 overlay ordered lookup, and L0 overlay scan amplification.
- `append_publish`: segment-only write-path benchmark with checksum verification enabled. It measures sorted and unsorted batch publish into fresh stores.
- `parameter_evolution`: segment-only cache-evolution benchmark with checksum verification enabled. It measures rebuild-vs-L0 behavior for middle inserts and repeated axis changes.
- `physical_format`: segment-only trusted-open, short/long-key sparse-routing, and segment-sizing benchmark. One profile-driven sizing suite holds block geometry constant while comparing segment file count, open, ordered reads, append publication, and forced-normalization amplification across flush thresholds and 16-byte through 16-KiB K/V profiles. Append publication prepares the empty store and `WriteBatch` outside the measured routine, so it times only commit processing and durable publication rather than setup or temporary-directory cleanup.
- `compression`: segment-only value-payload compression benchmark, available with `--features value-compression-lz4,value-compression-zstd`. It compares uncompressed stores against LZ4-created and Zstandard-level-1-created stores using the default writer-side compression policy, reports store bytes, and measures ordered fetch, full iteration, and append publish.

All current segment variants keep block checksum verification enabled, including the cross-backend comparison. The suite therefore measures the cache-safe implementation rather than an unchecked upper bound.

## Dataset Shape

The benchmark uses `128B` fixed-width keys. The synthetic key layout has long stable prefixes, ordered parameter axes, and a repetition-like suffix, which is closer to canonical-encoded parameter sweeps than random short keys.

Value profiles are narrow within one namespace:

- `small = 64B ± 8B`
- `small_fixed = 64B`
- `medium = 1KB ± 128B`
- `large = 16KB ± 2KB`

`small_fixed` uses `CreateOptions::with_fixed_value_len`, so the segment backend exercises the fixed-value block layout automatically for that profile. The benchmark intentionally does not mix small, medium, and large values inside one namespace profile.

Each profile currently uses `16,384` records for the standard dataset. The parameter-evolution datasets are separate structured grids that repeatedly change active `x` and `y` axes.

## Baseline Backends

`fjall3` uses a workload-shaped, no-compression configuration derived from the surrounding project: large cache, higher cached-file limit, point-read-hit expectation, profile-sensitive data block tuning, and key-value separation for large values. Compression is omitted because benchmark values are synthetic bytes and the goal is to measure storage layout rather than compression ratio.

`redb` is included as a mature embedded copy-on-write B-tree baseline. It uses one raw byte table and no domain-specific adapter layer.

## Workloads

### `comparison_ordered_fetch`

Fetches the full ordered stream of known keys and touches every returned value. This is the primary hot-path comparison. Variants: `segment`, `fjall3`, and `redb`; the large profile additionally registers `segment_256k` and `segment_512k` to measure block-size sensitivity against the other backends in the same run.

### `comparison_clustered_sparse_ordered_fetch`

Fetches a clustered ordered 1/16 subset of known keys: each 256-record window contributes one contiguous 16-record burst. This keeps the sparse ratio comparable to the old evenly-spaced stress test while better modeling bursty parameter subsets where some blocks have multiple hits and others have none. Variants: `segment`, `fjall3`, and `redb`; the large profile also measures the 256 and 512 KiB segment variants to expose sparse read amplification.

### `comparison_iter_all`

Scans every visible record in order and touches every value. This represents export, migration, and full ordered readback. Variants: `segment`, `fjall3`, and `redb`; the large profile also measures 256 and 512 KiB segment blocks.

### `comparison_append_publish`

Publishes one sorted batch into a fresh store. This is the cleanest cross-backend write comparison for append-style cache materialization. Variants: `segment`, `fjall3`, and `redb`; the large profile also records the 256 and 512 KiB segment block policies.

### `comparison_axis_change_rounds`

Runs several cache-shaped rounds where the active Cartesian-product parameter set changes over time. Each round does ordered lookup, touches hit values, and commits only misses. This target is registered only for the `small` profile to keep the cross-backend matrix bounded.

### `ordered_fetch`

Segment-only dense ordered lookup with checksum verification enabled. This is a stable internal regression target for the default read path.

### `clustered_sparse_ordered_fetch`

Segment-only clustered sparse ordered lookup with checksum verification enabled. It sweeps block sizes `16K`, `32K`, `64K`, and `256K` to expose sparse-read amplification when hits have local bursts.

### `random_sparse_ordered_fetch`

Segment-only seeded random sparse ordered lookup with checksum verification enabled. It keeps the query stream sorted after sampling, but the selected keys are not evenly spaced. This tests a less regular sparse shape than the old every-16th-key stress case.

### `cold_first_touch_checksum`

Segment-only first-touch checksum benchmark for the `large` profile with 4,096 records and 256K blocks. Every Criterion iteration opens a fresh read-only store in setup, resetting process-local verification state without clearing the OS page cache. `all_miss` uses keys interleaved between stored keys inside block ranges, so it measures lookup validation without returning values; `all_hit` verifies the corresponding payloads and touches every value. Store-open time is excluded from the measured routine.

### `large_value_block_size`

Segment-only dense ordered lookup for the `large` profile with block sizes `16K`, `64K`, `256K`, and `512K`. This answers whether large-value performance is blocked by block sizing or by value movement and checksum cost.

### `large_value_fetch_api`

Segment-only dense ordered lookup for the `large` profile comparing the borrowed visitor path with the owned `fetch_many_ordered` path at `16K` and `512K` block sizes. This isolates API-level value materialization cost from physical block-size effects.

### `overlay_ordered_fetch`

Segment-only L0 overlay read benchmark. `main_only` stores every record in the main tier; `patch_1`, `patch_2`, `patch_4`, and `patch_8` store one eighth of the records in patch segments and verify that all stores expose the same logical records.

### `overlay_iter_all`

Segment-only L0 overlay scan benchmark. It uses the same logical layout as `overlay_ordered_fetch`, but scans every visible record through `visit_all`, exercising the merge cursor used when patch segments are present.

### `append_publish`

Segment-only publish benchmark for `sorted_batch` and `unsorted_batch`. The sorted variant models callers that already emit canonical key order; the unsorted variant includes backend sorting cost. `append_publish_many_segments` uses 4,096 small records and a 128-record flush threshold to produce exactly 32 segments, isolating segment-directory publication overhead from large-value IO.

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
