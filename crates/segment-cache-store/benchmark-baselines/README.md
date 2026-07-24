# Benchmark Baselines

Saved benchmark results. Each subdirectory is one identified run with the commit, host, feature set, and exact commands needed to reproduce it. The workloads and dataset shapes these runs exercise are defined in [docs/benchmark.md](../docs/benchmark.md).

## Reading Results

The benchmark should be read as a shape test: whether the backend stays in the right performance class, whether it wins on the workload it is designed for, and whether it preserves its space-efficiency advantage. Winning every microbenchmark is less important than keeping ordered cache replay, append publish, corruption-safe reads, and file-level stability coherent in one backend.

Local smoke runs are useful for rough shape and regression awareness, not as publication-grade measurements. Runs that compare against a preceding baseline use the same benchmark sources, target directory, data, Criterion settings, and host session, executed sequentially to avoid CPU and filesystem contention.

## Index

- [`b093515-m1`](b093515-m1/README.md): Apple M1 baseline for commit `b093515`, covering all five Criterion targets and the space-usage report.
- [`9f8b2e6-m1`](9f8b2e6-m1/README.md): targeted Apple M1 baseline for the read, publication, normalization, and compression optimizations in commit `9f8b2e6`.
- [`4cdc470-m1-large-blocks`](4cdc470-m1-large-blocks/README.md): same-run large-profile cross-backend comparison of 16, 256, and 512 KiB segment block policies.
- [`v1-format-20260713-m1`](v1-format-20260713-m1/README.md): focused physical-format comparison covering trusted open, prefix-compressed segment bytes, ordered reads, and compressed reads.
- [`relative-key-routing-20260713-m1`](relative-key-routing-20260713-m1/README.md): focused comparison after routing progressively strips validated segment and block prefixes. This is the latest focused read-routing baseline.
- [`segment-sizing-20260713-m1`](segment-sizing-20260713-m1/README.md): segment flush sweep across tiny, long-key, medium-value, and large-value profiles, covering file layout, open, ordered reads, append publication, and forced-normalization amplification.
- [`historical-prototype`](historical-prototype/README.md): local smoke-run results from the prototype benchmark suite before the target rename. Rough shape and regression awareness only.
