# Large-Block Cross-Backend Baseline `4cdc470-m1-large-blocks`

## Identity

- Library commit: `4cdc4702bf5952b15c647a344c12d0fa019b351d`
- Date: 2026-07-13
- Host: Apple M1, 16 GiB RAM, Darwin 25.5.0, APFS-backed temporary directories
- Rust: `rustc 1.97.0 (2d8144b78 2026-07-07)`
- Features: `--all-features`
- Results: 20 targeted large-profile cases in [criterion.tsv](criterion.tsv)

The comparison harness keeps the historical `segment` label for the 16 KiB target block size and adds `segment_256k` and `segment_512k` only for the large profile. All segment variants use the same dataset, RapidHash payload verification, borrowed visitor APIs, and storage implementation. Fjall uses its existing large-value K/V-separation tuning; redb uses the existing single-table configuration.

## Commands

The commands were run serially. The complete comparison established the same-run sparse, scan, append, and non-large results. The ordered group was repeated with 20 samples because the first 512 KiB run contained severe high outliers; the TSV records the repeat for all ordered variants.

```bash
cargo bench -p segment-cache-store --bench comparison --all-features -- --sample-size 10 --warm-up-time 1 --measurement-time 2 --noplot --save-baseline 4cdc470-m1-cross
cargo bench -p segment-cache-store --bench comparison --all-features -- large/comparison_ordered_fetch --sample-size 20 --warm-up-time 2 --measurement-time 3 --noplot --save-baseline 4cdc470-m1-cross-read
```

## Large Ordered Fetch

| backend | result | relative to 16 KiB segment |
| --- | ---: | ---: |
| segment, 16 KiB | 33.937 ms | baseline |
| segment, 256 KiB | 23.666 ms | -30.3% |
| segment, 512 KiB | 23.096 ms | -31.9% |
| fjall3 | 24.385 ms | -28.1% |
| redb | 16.400 ms | -51.7% |

The 512 KiB segment variant is 5.3% faster than fjall in this dense ordered workload, while redb remains 40.8% faster than the 512 KiB segment result. Larger blocks remove most per-block read and decode overhead for dense replay, but do not eliminate the remaining gap to redb.

## Large Clustered Sparse Fetch

| backend | result | relative to 16 KiB segment |
| --- | ---: | ---: |
| segment, 16 KiB | 2.070 ms | baseline |
| segment, 256 KiB | 2.598 ms | +25.5% |
| segment, 512 KiB | 3.553 ms | +71.7% |
| fjall3 | 1.293 ms | -37.5% |
| redb | 0.890 ms | -57.0% |

The smallest tested block is the best segment configuration for clustered sparse reads. Reading a complete 256 or 512 KiB block to return a short burst of large values increases read amplification, so one static large block size cannot optimize both dense replay and sparse lookup.

## Large Full Scan

| backend | result | relative to 16 KiB segment |
| --- | ---: | ---: |
| segment, 16 KiB | 29.709 ms | baseline |
| segment, 256 KiB | 23.538 ms | -20.8% |
| segment, 512 KiB | 22.933 ms | -22.8% |
| fjall3 | 15.263 ms | -48.6% |
| redb | 15.219 ms | -48.8% |

Larger blocks materially improve the segment scan, but the best 512 KiB result remains about 50% slower than fjall and redb. The residual scan gap therefore is not explained by block size alone; sequential read coalescing, read-ahead, or other per-block traversal costs remain candidates.

## Large Append Publish

| backend | result | 95% confidence interval |
| --- | ---: | ---: |
| segment, 16 KiB | 934.397 ms | 920.173-947.426 ms |
| segment, 256 KiB | 817.129 ms | 771.134-885.987 ms |
| segment, 512 KiB | 848.847 ms | 770.829-997.864 ms |
| fjall3 | 647.889 ms | 629.530-664.640 ms |
| redb | 588.851 ms | 580.919-602.530 ms |

The 256 KiB point estimate is 12.5% faster than the 16 KiB segment configuration, but both larger-block variants have wide filesystem-sensitive confidence intervals. This run shows a remaining append gap to fjall and redb, but it does not distinguish encoding cost from filesystem durability noise strongly enough to select 256 versus 512 KiB on write latency.

## Conclusion

Block size is a workload policy, not a store-wide performance fix. A 512 KiB target is best among these variants for dense ordered replay and full scan, while 16 KiB is best for clustered sparse lookup. The current default should not be changed globally from this result. The next optimization should either make block sizing workload-aware or reduce full-block read amplification through a read strategy that preserves the unified block representation.
