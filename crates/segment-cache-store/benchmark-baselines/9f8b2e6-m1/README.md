# Targeted Benchmark Baseline `9f8b2e6-m1`

## Identity

- Implementation commit: `9f8b2e6b`
- Comparison baseline: `b093515-m1`
- Date: 2026-07-12
- Host: Apple M1, 16 GiB RAM, Darwin 25.5.0, APFS-backed temporary directories
- Rust: `rustc 1.97.0 (2d8144b78 2026-07-07)`
- Features: `--all-features`
- Results: 30 targeted cases in [criterion.tsv](criterion.tsv)

This is a targeted optimization baseline, not a replacement for the complete 161-case `b093515-m1` baseline. It records the workloads directly affected by phased checksum verification, single-sweep ordered lookup, cursor key scratch, batched segment publication, bounded normalization, and reusable compression buffers. The TSV stores Criterion's slope estimate when available and otherwise its mean estimate, with 95% confidence bounds in nanoseconds.

## Commands

The benchmark commands were run serially to avoid CPU, page-cache, and filesystem contention.

```bash
cargo bench -p segment-cache-store --bench ordered_lookup --all-features -- large/cold_first_touch_checksum --sample-size 20 --warm-up-time 2 --measurement-time 3 --noplot --baseline b093515-m1
cargo bench -p segment-cache-store --bench ordered_lookup --all-features -- 'large/(ordered_fetch|clustered_sparse_ordered_fetch|random_sparse_ordered_fetch|large_value_fetch_api|large_value_block_size)' --sample-size 10 --warm-up-time 1 --measurement-time 2 --noplot --baseline b093515-m1
cargo bench -p segment-cache-store --bench ordered_lookup --all-features -- 'small/overlay_(ordered_fetch|iter_all)' --sample-size 10 --warm-up-time 1 --measurement-time 2 --noplot --baseline b093515-m1
cargo bench -p segment-cache-store --bench ordered_lookup --all-features -- small/overlay_ordered_fetch/patch_4 --sample-size 20 --warm-up-time 2 --measurement-time 3 --noplot --baseline b093515-m1
cargo bench -p segment-cache-store --bench append_publish --all-features -- small/append_publish_many_segments --sample-size 20 --warm-up-time 2 --measurement-time 3 --noplot --baseline b093515-m1
cargo bench -p segment-cache-store --bench compression --all-features -- compression_append_publish --sample-size 10 --warm-up-time 1 --measurement-time 2 --noplot --baseline b093515-m1
cargo bench -p segment-cache-store --bench parameter_evolution --all-features -- middle_insert_then_read --sample-size 10 --warm-up-time 1 --measurement-time 2 --noplot --baseline b093515-m1
```

## Read Path

| workload | `b093515-m1` | `9f8b2e6-m1` | point change |
| --- | ---: | ---: | ---: |
| cold checksum, all miss | 6.016 ms | 4.530 ms | -24.7% |
| cold checksum, all hit | 7.828 ms | 7.379 ms | -5.7% |
| large dense ordered fetch | 35.403 ms | 33.187 ms | -6.3% |
| clustered sparse, 16K block | 2.312 ms | 1.998 ms | -13.6% |
| clustered sparse, 32K block | 1.982 ms | 1.908 ms | -3.7% |
| clustered sparse, 64K block | 2.008 ms | 1.716 ms | -14.6% |
| clustered sparse, 256K block | 2.716 ms | 2.566 ms | -5.5% |
| random sparse, 16K block | 2.574 ms | 2.378 ms | -7.6% |
| random sparse, 32K block | 3.198 ms | 3.146 ms | -1.6% |
| random sparse, 64K block | 4.830 ms | 4.718 ms | -2.3% |
| random sparse, 256K block | 12.654 ms | 11.786 ms | -6.9% |
| overlay ordered, main only | 1.153 ms | 1.120 ms | -2.9% |
| overlay ordered, patch 8 | 1.727 ms | 1.686 ms | -2.3% |
| overlay scan, main only | 1.041 ms | 0.959 ms | -7.9% |
| overlay scan, patch 8 | 1.895 ms | 1.711 ms | -9.7% |

The cold all-miss result is the direct measurement of deferred payload verification: blocks still validate lookup metadata, but misses no longer compute a payload digest. The smaller all-hit improvement preserves payload verification while avoiding the duplicate ordered-key sweep. The initial 10-sample `overlay_ordered_fetch/patch_4` run was noisy; a dedicated 20-sample repeat found no statistically significant change, with a point estimate of 1.785 ms and a 1.687-1.994 ms confidence interval.

## Publication And Normalization

| workload | `b093515-m1` | `9f8b2e6-m1` | point change |
| --- | ---: | ---: | ---: |
| publish 32 segments | 307.580 ms | 184.805 ms | -39.9% |
| small rebuild then read | 99.403 ms | 87.355 ms | -12.1% |
| small L0 insert then read | 159.787 ms | 165.363 ms | +3.5% |

The 32-segment result isolates the one-directory-sync publication barrier and shows the intended reduction directly. Small L0 insertion and the medium/large variants had no statistically significant change, so bounded normalization should currently be treated as a memory-scaling improvement rather than a demonstrated latency improvement.

## Compression Publish

| entropy shape | mode | `b093515-m1` | `9f8b2e6-m1` | point change |
| --- | --- | ---: | ---: | ---: |
| random bytes | none | 254.956 ms | 281.498 ms | +10.4% |
| random bytes | LZ4 | 251.789 ms | 271.728 ms | +7.9% |
| random bytes | Zstd 1 | 277.852 ms | 281.038 ms | +1.1% |
| template noise | none | 244.232 ms | 265.787 ms | +8.8% |
| template noise | LZ4 | 265.549 ms | 278.905 ms | +5.0% |
| template noise | Zstd 1 | 257.759 ms | 233.578 ms | -9.4% |
| correlated series | none | 242.726 ms | 276.986 ms | +14.1% |
| correlated series | LZ4 | 323.332 ms | 306.477 ms | -5.2% |
| correlated series | Zstd 1 | 381.307 ms | 339.628 ms | -10.9% |
| repeated runs | none | 246.939 ms | 264.363 ms | +7.1% |
| repeated runs | LZ4 | 140.573 ms | 95.656 ms | -32.0% |
| repeated runs | Zstd 1 | 143.578 ms | 110.534 ms | -23.0% |

This target includes filesystem creation, writes, and durability syncs. Every `none` case was 7-14% slower than the older run, so the small regressions for incompressible LZ4 and the mixed template-noise result cannot be attributed cleanly to the codec change. The larger correlated-series and repeated-runs improvements remain directionally useful because compressed modes improved while their same-run uncompressed controls regressed. Use repeated runs or a future encoder-only benchmark before treating smaller compression deltas as regression gates.

## Interpretation

The strongest stable results are the cold all-miss checksum improvement, the 32-segment publication improvement, and the scan improvements from replacing full-block key materialization with one cursor key scratch. Dense and sparse ordered reads are either unchanged or faster across the measured shapes. Filesystem-heavy compression and large normalization cases remain noisy and should be compared by confidence interval and same-run controls rather than point estimate alone.
