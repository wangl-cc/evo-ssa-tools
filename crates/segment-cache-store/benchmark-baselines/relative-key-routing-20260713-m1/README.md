# Relative-Key Routing Baseline

## Identity

- Before: `f9ffb07` (`refactor(segment-cache-store): finalize v1 segment format`)
- After: working tree based on `f9ffb07`
- Date: 2026-07-13
- Host: Apple M1, 16 GiB RAM, Darwin 25.5.0, APFS-backed temporary directories
- Features: `--all-features`
- Rust: `rustc 1.97.0 (2d8144b78 2026-07-07)`

The before values are the saved v1-format measurements from the same host and date. The after values use the same benchmark sources, data, and Criterion settings. Benchmarks were run sequentially to avoid CPU and filesystem contention.

The after implementation compares a full key only while selecting a segment. Segment-index routing then uses a segment-relative key, and record lookup uses only the suffix after the selected block's common prefix. Loaded blocks no longer allocate or copy a combined segment-plus-block prefix.

## Commands

```bash
cargo bench -p segment-cache-store --bench physical_format --all-features -- --sample-size 20 --warm-up-time 2 --measurement-time 3 --noplot
cargo bench -p segment-cache-store --bench ordered_lookup --all-features -- 'large/ordered_fetch/default_block|large/random_sparse_ordered_fetch/block_16k|small/overlay_ordered_fetch/main_only|small/overlay_ordered_fetch/patch_8' --sample-size 20 --warm-up-time 2 --measurement-time 3 --noplot
cargo bench -p segment-cache-store --bench compression --all-features -- 'value_128k/repeated_runs/compression_(ordered_fetch|iter_all)' --sample-size 10 --warm-up-time 1 --measurement-time 1 --noplot
```

## Routing

| workload | before | after | change |
| --- | ---: | ---: | ---: |
| trusted open, 32 MiB segment | 111.33 us | 111.71 us | +0.34% |
| sparse routing, 16-byte keys | 380.67 us | 277.97 us | -26.98% |
| sparse routing, 512-byte keys | 547.08 us | 292.92 us | -46.46% |

Removing the loaded-block prefix allocation and comparing only the key part owned by each routing layer materially improves both key widths. Trusted open is unchanged, as expected.

## Existing Read Workloads

| workload | before | after | change |
| --- | ---: | ---: | ---: |
| large dense ordered fetch | 34.646 ms | 33.960 ms | -1.98% |
| large random sparse, 16 KiB blocks | 2.388 ms | 2.355 ms | -1.37% |
| small main-only ordered fetch | 1.160 ms | 950.92 us | -18.01% |
| small ordered fetch with 8 patches | 1.772 ms | 1.435 ms | -19.03% |

Compared with the pre-v1-format historical points, main-only and patch-8 are now 13.76% and 13.22% faster. Large dense and sparse remain 2.08% and 1.11% slower, respectively; those paths are dominated more by value IO and checksum work, and the remaining differences are small enough to require a longer run before treating them as a regression.

## Compression

| repeated-runs workload | before | after | change |
| --- | ---: | ---: | ---: |
| ordered, uncompressed | 6.072 ms | 6.059 ms | -0.21% |
| ordered, LZ4 | 15.571 ms | 15.661 ms | +0.58% |
| ordered, Zstd level 1 | 16.304 ms | 16.403 ms | +0.61% |
| scan, uncompressed | 6.004 ms | 5.988 ms | -0.27% |
| scan, LZ4 | 16.089 ms | 15.455 ms | -3.94% |
| scan, Zstd level 1 | 16.176 ms | 16.231 ms | +0.34% |

The compression measurements use a one-second window and show no systematic regression. The LZ4 scan point improved in this run, but the short measurement is sufficient only to conclude that direct prefix reconstruction did not make scans slower.

## Conclusion

Progressive relative-key routing removes the known long-prefix comparison defect and more than recovers the previous small-value ordered-read regression. It also simplifies the ownership model: segment and block range validation establish prefix membership once, query types carry only the remaining key bytes, and scan cursors reconstruct one key at a time into reusable scratch.
