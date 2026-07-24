# Historical Prototype Snapshot

## Identity

Local smoke-run results from the prototype benchmark suite before the target rename. These predate the per-run baseline format (no recorded commit, host, or Criterion settings beyond the commands quoted below). They are useful for rough shape and regression awareness, not as publication-grade measurements.

## Ordered fetch

| namespace | segment | fjall3 | redb |
| --- | ---: | ---: | ---: |
| small | 1.24-1.27 ms | 6.25-6.31 ms | 3.89-4.27 ms |
| small_fixed | 1.04-1.05 ms | 6.22-6.46 ms | 3.47-3.48 ms |
| medium | 4.87-5.11 ms | 6.54-6.58 ms | 5.14-5.18 ms |
| large | 61.89-63.39 ms | 26.65-27.14 ms | 22.64-25.75 ms |

## Large read baseline before K/V checksum split

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

## Large sparse after K/V checksum split

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

## Large reads after process-local verification reuse

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

## Append publish

| namespace | segment sorted | fjall3 | redb |
| --- | ---: | ---: | ---: |
| small | 64.29-66.92 ms | 119.88-124.82 ms | 73.79-75.89 ms |
| small_fixed | 64.63-67.93 ms | 120.10-125.13 ms | 75.41-79.90 ms |
| medium | 89.38-91.70 ms | 136.55-142.92 ms | 117.88-123.65 ms |
| large | 819.94-909.09 ms | 1.18-1.29 s | 752.32-816.80 ms |

## Space amplification

| namespace | logical bytes | segment | fixed layout | fjall3 | redb |
| --- | ---: | ---: | ---: | ---: | ---: |
| small | 3,145,563 | 1.077x | n/a | 21.336x | 2.679x |
| small_fixed | 3,145,728 | 1.077x | 1.035x | 21.334x | 2.678x |
| medium | 18,870,633 | 1.052x | n/a | 3.556x | 1.785x |
| large | 270,504,438 | 1.037x | n/a | 1.253x | 1.992x |
