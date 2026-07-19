# Benchmark Baseline `b093515-m1`

## Identity

- Commit: `b09351591e44a74dfe5d4eb426845b58f8954f53`
- Date: 2026-07-12
- Host: Apple M1, 16 GiB RAM, Darwin 25.5.0, APFS-backed temporary directories
- Rust: `rustc 1.97.0 (2d8144b78 2026-07-07)`
- Cargo: `cargo 1.97.0 (c980f4866 2026-06-30)`
- Features: `--all-features`
- Criterion baseline: `b093515-m1`
- Results: 161 benchmark cases in [criterion.tsv](criterion.tsv), plus repeated large-append measurements below

The TSV records Criterion's slope estimate when available and otherwise its mean estimate, together with the 95% confidence interval. All time values are nanoseconds. The ignored local Criterion data remains under `crates/segment-cache-store/target/criterion/**/b093515-m1` for direct comparisons in this worktree.

## Commands

The benchmark targets were run serially to avoid CPU, page-cache, and filesystem contention.

```bash
cargo bench -p segment-cache-store --bench ordered_lookup --all-features -- --sample-size 20 --warm-up-time 2 --measurement-time 3 --noplot --save-baseline b093515-m1
cargo bench -p segment-cache-store --bench ordered_lookup --all-features -- large/cold_first_touch_checksum --sample-size 20 --warm-up-time 2 --measurement-time 3 --noplot --save-baseline b093515-m1
cargo bench -p segment-cache-store --bench append_publish --all-features -- --sample-size 20 --warm-up-time 2 --measurement-time 3 --noplot --save-baseline b093515-m1
cargo bench -p segment-cache-store --bench append_publish --all-features -- small/append_publish_many_segments --sample-size 20 --warm-up-time 2 --measurement-time 3 --noplot --save-baseline b093515-m1
cargo bench -p segment-cache-store --bench parameter_evolution --all-features -- --sample-size 10 --warm-up-time 1 --measurement-time 2 --noplot --save-baseline b093515-m1
cargo bench -p segment-cache-store --bench comparison --all-features -- --sample-size 10 --warm-up-time 1 --measurement-time 2 --noplot --save-baseline b093515-m1
cargo bench -p segment-cache-store --bench compression --all-features -- --sample-size 10 --warm-up-time 1 --measurement-time 2 --noplot --save-baseline b093515-m1
cargo run -p segment-cache-store --example space_usage --all-features --offline

# Diagnose and replace unstable large-append samples.
cargo bench -p segment-cache-store --bench append_publish --all-features -- large/append_publish --sample-size 10 --warm-up-time 1 --measurement-time 2 --noplot --baseline b093515-m1
cargo bench -p segment-cache-store --bench append_publish --all-features -- large/append_publish --sample-size 20 --warm-up-time 2 --measurement-time 3 --noplot --save-baseline b093515-m1
```

For an immediate comparison while the ignored Criterion baseline is present, replace `--save-baseline b093515-m1` with `--baseline b093515-m1`. If the local baseline has been deleted, use the committed TSV for historical comparison or check out the baseline commit and rerun the commands above.

## Segment Read Path

| profile | dense ordered fetch |
| --- | ---: |
| small | 1.142 ms |
| small fixed | 0.900 ms |
| medium | 3.188 ms |
| large | 35.403 ms |

Large-value routing and materialization:

| workload | 16K | 32K | 64K | 256K | 512K |
| --- | ---: | ---: | ---: | ---: | ---: |
| clustered sparse | 2.312 ms | 1.982 ms | 2.008 ms | 2.716 ms | n/a |
| random sparse | 2.574 ms | 3.198 ms | 4.830 ms | 12.654 ms | n/a |
| borrowed dense | 34.727 ms | n/a | n/a | n/a | 23.665 ms |
| owned dense | 48.469 ms | n/a | n/a | n/a | 39.082 ms |

Small-value overlay amplification:

| workload | main only | 8 patch segments |
| --- | ---: | ---: |
| ordered fetch | 1.153 ms | 1.727 ms |
| full iteration | 1.041 ms | 1.895 ms |

Cold process-local checksum verification with 4,096 approximately 16 KiB values and 256K blocks:

| workload | result |
| --- | ---: |
| all miss | 6.016 ms |
| all hit | 7.828 ms |

Each iteration opens a fresh read-only store in Criterion setup, so the process-local block verification state starts empty while the OS page cache remains warm. Interleaved missing keys fall between stored keys inside block ranges, forcing block metadata lookup without returning values. This isolates checksum work from cold physical IO and gives the two-phase lookup/payload verification change a meaningful before/after measurement.

## Publication And Evolution

| profile | sorted append | unsorted append | rebuild then read | L0 insert then read |
| --- | ---: | ---: | ---: | ---: |
| small | 67.539 ms | 77.411 ms | 99.403 ms | 159.787 ms |
| small fixed | 69.793 ms | 77.251 ms | 105.830 ms | 151.880 ms |
| medium | 115.383 ms | 123.903 ms | 161.216 ms | 176.256 ms |
| large | 1.117 s | 1.482 s | 1.760 s | 1.856 s |

The dedicated multi-segment publication case writes 4,096 small records as exactly 32 segments and takes 307.580 ms [298.784, 316.057]. It isolates the cost of syncing the segment directory once per published file in the baseline implementation.

The large append ordering is not a trustworthy indication that sorting helps or hurts. Three back-to-back runs reversed the apparent ordering, while the independent comparison target measured segment append at 1.056 s:

| run | samples | sorted | unsorted |
| --- | ---: | ---: | ---: |
| initial baseline | 20 | 1.238 s [1.182, 1.298] | 1.027 s [1.004, 1.057] |
| comparison repeat | 10 | 1.040 s [1.019, 1.060] | 1.005 s [0.998, 1.011] |
| saved repeat | 20 | 1.117 s [1.035, 1.202] | 1.482 s [1.400, 1.566] |

Treat approximately 1.0-1.6 s as the current filesystem-noise range. The final 20-sample repeat is the value stored in Criterion and the TSV, but future writer work should improve benchmark isolation or aggregate multiple runs before using this case as a regression gate.

The small-profile axis-change workload completed in 114.539 ms with 15,616 queries, 9,088 hits, 6,528 misses, rewrite amplification 1.00, no retired segments, and five published segments.

## Cross-Backend Shape

Dense ordered fetch:

| profile | segment | fjall3 | redb |
| --- | ---: | ---: | ---: |
| small | 1.169 ms | 5.914 ms | 3.768 ms |
| small fixed | 0.902 ms | 5.661 ms | 3.600 ms |
| medium | 3.293 ms | 6.585 ms | 5.620 ms |
| large | 34.390 ms | 23.562 ms | 18.456 ms |

Full iteration:

| profile | segment | fjall3 | redb |
| --- | ---: | ---: | ---: |
| small | 0.981 ms | 1.583 ms | 0.739 ms |
| small fixed | 0.634 ms | 1.380 ms | 0.652 ms |
| medium | 2.744 ms | 1.956 ms | 2.130 ms |
| large | 39.826 ms | 16.740 ms | 16.767 ms |

The segment store is strongest for small and medium ordered cache replay. Large-value dense reads and scans remain the clearest optimization target. Cross-backend values are workload-shaped comparisons, not generic database rankings.

## Compression

Store bytes divided by raw value bytes:

| entropy shape | none | LZ4 | Zstd level 1 |
| --- | ---: | ---: | ---: |
| random bytes | 1.002x | 1.002x | 1.002x |
| template noise | 1.002x | 0.594x | 0.535x |
| correlated series | 1.002x | 0.548x | 0.396x |
| repeated runs | 1.002x | 0.006x | 0.007x |

Ordered fetch for 512 values of approximately 128 KiB:

| entropy shape | none | LZ4 | Zstd level 1 |
| --- | ---: | ---: | ---: |
| random bytes | 6.125 ms | 6.140 ms | 6.086 ms |
| template noise | 6.082 ms | 9.679 ms | 12.764 ms |
| correlated series | 6.259 ms | 24.138 ms | 66.676 ms |
| repeated runs | 6.222 ms | 15.951 ms | 16.610 ms |

The policy correctly stores incompressible random data raw. Compression provides major space savings for structured values but can add substantial read CPU, especially Zstd on correlated numeric series. Repeated-run publication is faster with compression because the reduction in bytes written outweighs codec cost.

## Space Usage

| profile | logical bytes | segment | fixed segment | fjall3 | redb |
| --- | ---: | ---: | ---: | ---: | ---: |
| small | 3,145,563 | 0.642x | n/a | 21.336x | 2.679x |
| small fixed | 3,145,728 | 0.642x | 0.623x | 21.334x | 2.678x |
| medium | 18,870,633 | 0.945x | n/a | 3.556x | 1.785x |
| large | 270,504,438 | 1.010x | n/a | 1.256x | 1.992x |

Ratios below one are possible because logical bytes count full 128-byte keys while the segment format stores shared key prefixes once.

## Interpretation Limits

These are predominantly warm, in-process Criterion measurements on an interactive development machine, not cold-cache or publication-grade results. The explicitly named first-touch checksum case resets process-local verification without clearing the OS page cache. Several microsecond workloads reported 10-30% outliers, and filesystem-heavy workloads showed materially wider run-to-run variation. Compare future changes on the same host with the same commands, inspect confidence intervals rather than only point estimates, and rerun any apparent regression near the observed noise floor.

The storage implementation remained at the recorded commit throughout the baseline run. The first-touch benchmark fixture was added afterward without changing library code, then measured against the same implementation. Concurrent documentation-only changes do not affect the compiled artifacts and are not part of this baseline.
