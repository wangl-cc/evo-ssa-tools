# V1 Segment Format Baseline

## Identity

- Before: `d9c2d28` (`test(segment-cache-store): compare large block policies`)
- After: working tree based on `d9c2d28`, before its final commit
- Date: 2026-07-13
- Host: Apple M1, 16 GiB RAM, Darwin 25.5.0, APFS-backed temporary directories
- Features: `--all-features`
- Rust: `rustc 1.97.0 (2d8144b78 2026-07-07)`

The before and after runs used the same benchmark sources, target directory, data, Criterion settings, and host session. Benchmarks were run sequentially to avoid CPU and filesystem contention. The after implementation stores a segment-wide key prefix once, stores block keys relative to it, trusts managed local files during ordinary open, and uses complete BLAKE3 content verification only at the explicit single-segment verification boundary.

## Commands

```bash
cargo bench -p segment-cache-store --bench physical_format --all-features -- --sample-size 20 --warm-up-time 2 --measurement-time 3 --noplot
cargo bench -p segment-cache-store --bench ordered_lookup --all-features -- 'large/ordered_fetch/default_block|large/random_sparse_ordered_fetch/block_16k|small/overlay_ordered_fetch/main_only|small/overlay_ordered_fetch/patch_8' --sample-size 20 --warm-up-time 2 --measurement-time 3 --noplot
cargo bench -p segment-cache-store --bench compression --all-features -- 'value_128k/repeated_runs/compression_(ordered_fetch|iter_all)' --sample-size 10 --warm-up-time 1 --measurement-time 1 --noplot
```

## Physical Format

| workload | before | after | change |
| --- | ---: | ---: | ---: |
| trusted open, 32 MiB segment | 44.636 ms | 111.33 us | -99.75%, about 401x faster |
| sparse routing, 16-byte keys | 379.52 us | 380.67 us | +0.30% |
| sparse routing, 512-byte keys | 565.03 us | 547.08 us | -3.18% |

| dataset | before segment bytes | after segment bytes | change |
| --- | ---: | ---: | ---: |
| open, 128-byte keys and 32 MiB values | 33,605,824 | 33,571,655 | -0.10% |
| routing, 16-byte keys | 1,149,820 | 1,147,844 | -0.17% |
| routing, 512-byte keys | 1,225,080 | 1,149,828 | -6.14% |

The first composite-key implementation compared each prefix byte through an iterator and made 512-byte routing about nine times slower. Split slice comparison removed that defect. Materializing one complete block prefix per loaded block, while keeping the runtime index and on-disk footer compact, restored 16-byte routing to parity and retained a small long-key improvement.

## Existing Read Workloads

| workload | before | after | change |
| --- | ---: | ---: | ---: |
| large dense ordered fetch | 33.268 ms | 34.646 ms | +4.14% |
| large random sparse, 16 KiB blocks | 2.329 ms | 2.388 ms | +2.52% |
| small main-only ordered fetch | 1.103 ms | 1.160 ms | +5.19% |
| small ordered fetch with 8 patches | 1.653 ms | 1.772 ms | +7.17% |

The remaining regression is concentrated in small-value ordered reads where key routing is a larger fraction of total work. Large-value and sparse paths move less because value IO and checksum work dominate. This is the measured cost of the segment-relative runtime representation after removing the initial pathological comparison implementation.

## Compression

| repeated-runs workload | before | after | change |
| --- | ---: | ---: | ---: |
| ordered, uncompressed | 6.394 ms | 6.072 ms | -5.03% |
| ordered, LZ4 | 15.803 ms | 15.571 ms | -1.47% |
| ordered, Zstd level 1 | 16.257 ms | 16.304 ms | +0.29% |
| scan, uncompressed | 5.916 ms | 6.004 ms | +1.49% |
| scan, LZ4 | 15.713 ms | 16.089 ms | +2.39% |
| scan, Zstd level 1 | 16.200 ms | 16.176 ms | -0.15% |

The compressed read differences are small relative to the short one-second measurement window and show no systematic decode regression. Prefix deduplication is more visible in space usage when the 16 KiB split target produces many single-record blocks: LZ4 bytes fell from 431,908 to 352,487 (-18.39%), and Zstd bytes fell from 490,778 to 411,357 (-16.18%).

## Conclusion

The local-trust open path removes the previous whole-file scan and is the dominant win. Segment-prefix encoding materially improves long-key and highly compressed file size without slowing the focused long-key routing case. The accepted tradeoff is a measured 2.5-7.2% regression in existing ordered read workloads; further work should target composite segment-index routing or reusable loaded-block prefix storage only if those small-value paths become a product bottleneck.
