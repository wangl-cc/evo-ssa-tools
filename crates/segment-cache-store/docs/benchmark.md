# Benchmarking

## Purpose

The benchmark suite is designed to evaluate `segment-cache-store` as a storage backend, not as a domain-specific scientific workload.

That means the benchmark models:

- key width
- value size distribution
- ordered access locality
- hit and miss shape
- append-only publish behavior

It does not model the scientific meaning of a value. Once the value is serialized, the storage layer only cares about the physical byte behavior.

This is intentional. For this crate, the relevant distinction is not whether a namespace contains "final state", "dynamics", or "tree" objects. The relevant distinction is whether one namespace behaves like:

- small, narrow-distribution values
- small, truly fixed-length values
- medium, narrow-distribution values
- large, narrow-distribution values

The benchmark therefore uses storage-shaped profiles rather than application-level type names.

## What the Benchmark Tries to Approximate

The target workload has these storage-level characteristics:

- wide fixed-width keys
- one namespace tends to have a narrow value-size distribution
- reads are usually ordered batch reads
- writes are append-only publishes
- downstream analysis often replays previously materialized ordered key streams

The benchmark therefore groups scenarios by storage shape rather than by application-level result type.

For `fjall3`, the benchmark does not use a pure out-of-the-box default configuration anymore. It now uses a workload-shaped, no-compression configuration derived from `/Users/loong/Repos/Projects/pheno-geno/pybinding/src/db.rs`:

- database cache size
- cached file limit
- point-read-hit expectation
- profile-sensitive block size and hash-ratio tuning
- `KvSeparation` for the large-value profile

The compression-related options from the production configuration are intentionally omitted. The benchmark values are synthetic byte payloads, and compression would mix codec/compression behavior into a storage-layout benchmark.

`redb` is included as a second general-purpose baseline. It represents a mature embedded copy-on-write B-tree with transactional semantics and a single-file physical layout. The benchmark uses raw byte keys and values in one table and does not add any domain-specific adapter layer around it.

## Dataset Shape

The benchmark lives in [benches/workload.rs](../benches/workload.rs).

### Keys

The benchmark uses `128B` fixed-width keys.

That is intentionally wider than a toy key and is meant to be closer to real canonical-encoded parameter keys, where many fields are fixed or slow-moving and only later fields change between nearby entries.

The synthetic key layout is structured so that:

- nearby keys share long prefixes
- repetition-like variation appears late in the key
- the overall key stream remains globally ordered
- the sparse ordered subset takes one key from every 16-key repetition group

This is more realistic for computation-cache workloads than random short keys.

### Values

There are four value-size profiles:

- `small = 64B ± 8B`
- `small_fixed = 64B`
- `medium = 1KB ± 128B`
- `large = 16KB ± 2KB`

This matches the intended storage assumption that one namespace usually contains values of roughly similar size.

The benchmark intentionally does not use a mixed cross-namespace value distribution inside one profile.

`small_fixed` additionally exercises `CreateOptions::with_fixed_value_len`, which removes per-record value lengths and value offsets from segment blocks. It is only appropriate when the whole store has a stable fixed value length.

The current `fjall3` benchmark adapter maps these profiles onto two no-compression tuning classes:

- `small`, `small_fixed`, and `medium` use the small-value `fjall3` tuning
- `large` uses the large-value `fjall3` tuning

That matches the current reference implementation more closely than a one-size-fits-all keyspace configuration.

### Record Count

Each benchmark profile currently uses `16,384` records.

This is large enough to produce multiple blocks and multiple segments while remaining light enough for quick local iteration.

## Workloads

### `ordered_contains`

Checks a fully ordered stream of known keys and counts hits.

This represents:

- ordered membership checks
- cache preflight hit checks
- workloads that want hit/miss structure without needing payload bytes

Why it matters:

- it isolates ordered key traversal overhead
- it shows whether the backend preserves locality well

### `ordered_fetch`

Fetches a fully ordered stream of known keys and aggregates payload lengths.

This is the main hot-path workload.

It represents:

- replaying a cached parameter sweep
- warm-cache downstream analysis
- repeated ordered reads through one namespace

The benchmark includes:

- `segment`: fresh ordered read through `Store`
- `segment_no_crc`: ordered read with block checksum verification disabled, isolating the CRC cost
- `segment_block_{64,256,512,1024}k` and matching `..._no_crc` variants on the `large` profile, sweeping the segment block size
- `segment_fixed_layout` and `segment_fixed_layout_no_crc` on the `small_fixed` profile, exercising the fixed-value-length layout
- `fjall3`
- `redb`

Why it matters:

- ordered batch fetch is the primary design target for this backend
- this is the most important direct comparison against general KV engines

The segment variants are intended to isolate costs in the same store-level ordered batch path:

- `segment` uses the store-level ordered batch API directly
- `segment_no_crc` disables block checksum verification
- block-size and fixed-layout variants attribute physical layout costs

A separate reused-session benchmark is intentionally not registered here. The store-level batch API already creates and uses an `OrderedLookup` for the duration of each full ordered stream, so benchmarking one reused session over the same full stream mostly measures the same traversal and can be misleading unless the workload is specifically windowed across multiple adjacent calls.

### `sparse_ordered_fetch`

Fetches an ordered 1/16 subset of known keys by taking one key from each repetition group.

This represents:

- selecting one replicate or subcase from many cached parameter groups
- ordered but sparse reuse
- the read-amplification tradeoff of larger segment blocks

The benchmark includes `segment_block_{16,32,64,256}k`, `fjall3`, and `redb`. It intentionally does not register no-CRC variants, because this workload is primarily about block size and sparse read amplification rather than isolating checksum cost.

Why it matters:

- full ordered fetch can make larger blocks look better than they are for sparse subsets
- sparse ordered fetch helps choose a safe default block size for mixed dense/sparse read patterns

### `append_commit`

Publishes one full batch into a fresh store.

It represents:

- miss-then-compute-then-publish
- one namespace flushing a completed batch of new cache entries

The benchmark includes:

- `segment_sorted`: caller already knows keys are sorted
- `segment_unsorted`: backend sorts before publish
- `fjall3`
- `redb`

Why it matters:

- immutable segment publish is expected to be a strong path for this design

### `iter_all`

Scans every record in order.

It represents:

- export
- migration
- full ordered readback

Why it matters:

- iteration is a real requirement, but not the primary optimization target

### `two_pass_cache_shape`

Loads the store once, then runs one ordered read stream with both hits and misses.

It represents:

- a cache-shaped replay pass after data has already been published
- ordered downstream analysis with partial reuse

Why it matters:

- this is the scenario where append-only publish plus ordered replay should show the strongest overall advantage

## Additional Space Metric

Space amplification is treated as a first-class metric.

The measurement script lives in [examples/space_usage.rs](../examples/space_usage.rs).

Run it with:

```bash
cargo run -p segment-cache-store --example space_usage --offline
```

This prints, for each namespace profile:

- logical bytes
- actual segment-cache-store directory size
- actual `fjall3` directory size
- actual `redb` directory size
- amplification factor for each backend

## Commands

Run the full Criterion suite:

```bash
cargo bench -p segment-cache-store --bench workload
```

Run one Criterion subgroup:

```bash
cargo bench -p segment-cache-store --bench workload -- ordered_fetch
```

Measure space amplification:

```bash
cargo run -p segment-cache-store --example space_usage --offline
```

## Latest Snapshot

The numbers below are local smoke-run results, not publication-grade measurements. They are still useful for relative shape and regression tracking.

Benchmark command:

```bash
cargo bench -p segment-cache-store --bench workload --offline
```

Space command:

```bash
cargo run -p segment-cache-store --example space_usage --offline
```

### Ordered fetch

Representative recent snapshot:

| namespace | segment | segment reused lookup | fjall3 | redb |
| --- | ---: | ---: | ---: | ---: |
| small | 1.24-1.27 ms | 1.25-1.27 ms | 6.25-6.31 ms | 3.89-4.27 ms |
| small_fixed | 1.04-1.05 ms | 1.04-1.08 ms | 6.22-6.46 ms | 3.47-3.48 ms |
| small_fixed fixed layout | 0.99-1.02 ms | 0.99-1.01 ms | n/a | n/a |
| medium | 4.87-5.11 ms | 4.83-4.98 ms | 6.54-6.58 ms | 5.14-5.18 ms |
| large | 61.89-63.39 ms | 60.79-67.02 ms | 26.65-27.14 ms | 22.64-25.75 ms |

Interpretation:

- the backend is firmly in the same order of magnitude as tuned `fjall3` and `redb`
- small-value ordered fetch remains materially faster than both general-purpose baselines
- medium-value ordered fetch is close to `redb` and faster than tuned `fjall3`
- large-value ordered fetch is behind both `redb` and no-compression `fjall3`
- fixed layout gives the best result for the fixed-small profile, but the regular layout is already close

### Append commit

Representative recent snapshot:

| namespace | segment sorted | fjall3 | redb |
| --- | ---: | ---: | ---: |
| small | 64.29-66.92 ms | 119.88-124.82 ms | 73.79-75.89 ms |
| small_fixed | 64.63-67.93 ms | 120.10-125.13 ms | 75.41-79.90 ms |
| small_fixed fixed layout | 60.97-64.65 ms | n/a | n/a |
| medium | 89.38-91.70 ms | 136.55-142.92 ms | 117.88-123.65 ms |
| large | 819.94-909.09 ms | 1.18-1.29 s | 752.32-816.80 ms |

Interpretation:

- immutable append publish is one of the strongest paths for this backend
- segment publish consistently beats `fjall3` for small and medium profiles
- `redb` is a stronger write baseline than `fjall3` in this run, especially for large values
- large append is now close to `redb` and faster than no-compression `fjall3`, but it is much less dominant than the small-value cases

### Two-pass cache-shaped workload

Representative recent snapshot:

| namespace | segment | fjall3 | redb |
| --- | ---: | ---: | ---: |
| small | 60.17-62.66 ms | 121.99-125.59 ms | 76.70-78.99 ms |
| small_fixed | 69.67-72.02 ms | 123.93-127.32 ms | 75.48-77.09 ms |
| small_fixed fixed layout | 70.90-73.05 ms | n/a | n/a |
| medium | 88.94-90.81 ms | 144.20-150.63 ms | 123.46-132.35 ms |
| large | 865.76-926.69 ms | 1.34-1.52 s | 670.45-715.97 ms |

Interpretation:

- the backend does best on the workload shape it was designed for
- small and medium two-pass workloads remain strong for the segment store
- large two-pass is a weak spot for the current segment implementation; `redb` is faster in this run

### Space amplification

Recent `space_usage` snapshot:

| namespace | logical bytes | segment | fixed layout | fjall3 | redb |
| --- | ---: | ---: | ---: | ---: | ---: |
| small | 3,145,563 | 1.077x | n/a | 21.336x | 2.679x |
| small_fixed | 3,145,728 | 1.077x | 1.035x | 21.334x | 2.678x |
| medium | 18,870,633 | 1.052x | n/a | 3.556x | 1.785x |
| large | 270,504,438 | 1.037x | n/a | 1.253x | 1.992x |

Interpretation:

- for narrow-distribution namespaces, the segment layout is very space-efficient
- the oversized-block padding change was especially important for large-value namespaces
- `redb` has much lower amplification than `fjall3` for small and medium values, but higher amplification than both segment and `fjall3` for the current large-value profile

## How to Read the Results

The most important takeaways are:

- if the workload is dominated by small or medium ordered replay, `segment-cache-store` is competitive and often stronger in end-to-end cache-shaped scenarios
- if the workload is dominated by large-value ordered fetch, both `redb` and tuned no-compression `fjall3` currently beat the segment backend
- append-only publish is a clear strength for small and medium profiles; large-value append is competitive but not dominant against `redb`
- space amplification is much lower than the compared general-purpose backends for small and medium namespaces, and still best for the current large profile

The benchmark should therefore be read as a shape test:

- does the backend stay within the right performance class?
- does it win on the workload it is designed for?
- does it preserve its space-efficiency advantage?

Those questions matter more here than winning every microbenchmark.
