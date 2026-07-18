# Segment Sizing Baseline

## Identity

- Layout and read commit: `c95e88e` (`perf(segment-cache-store): strip validated key prefixes`)
- Corrected append commit: working tree based on `d14ae6d` (`feat(segment-cache-store): enforce unique-key merge semantics`)
- Date: 2026-07-13 for layout and reads; 2026-07-17 for corrected append publication
- Host: Apple M1, 16 GiB RAM, Darwin 25.5.0, APFS-backed temporary directories
- Features: `--all-features`
- Rust: `rustc 1.97.0 (2d8144b78 2026-07-07)`

The benchmark holds the block target at its 16 KiB default and varies only segment flush policy. Value compression is disabled and block checksum verification remains enabled. The first sweep isolates the record threshold with 131,072 records using 128-byte keys and 64-byte values. The second sweep varies K/V geometry to determine when the record threshold or byte threshold controls segment size. Sparse lookups are deterministic stride samples, not random samples.

## Command

```bash
cargo bench -p segment-cache-store --bench physical_format --all-features -- 'segment_sizing/' --sample-size 20 --warm-up-time 2 --measurement-time 3 --noplot
cargo bench -p segment-cache-store --bench physical_format --all-features -- 'segment_size_profiles/' --sample-size 10 --warm-up-time 1 --measurement-time 2 --noplot
cargo bench -p segment-cache-store --bench physical_format --all-features -- append_publish --sample-size 20 --warm-up-time 2 --measurement-time 3 --noplot
```

The corrected append benchmark prepares the empty store and `WriteBatch` outside Criterion's measured routine. It measures `commit_batch_with_options` processing and durable publication, excluding store creation, batch construction, and temporary-directory cleanup.

## Record-Threshold Sweep

This sweep uses 131,072 records with 128-byte keys and 64-byte values. Every policy writes the same logical records and approximately 9.2 MB of segment files; only the number and average size of immutable segment files change.

### Layout

| record threshold | byte threshold | segment files | average segment bytes |
| ---: | ---: | ---: | ---: |
| 4,096 | 8 MiB | 32 | 287,674 |
| 16,384 | 16 MiB | 8 | 1,150,092 |
| 32,768 | 32 MiB | 4 | 2,300,126 |
| 65,536 | 32 MiB | 2 | 4,600,817 |
| 131,072 | 32 MiB | 1 | 9,201,789 |

The record threshold controls every variant in this small-value dataset. No variant reaches its byte threshold, so these results do not select a new default byte threshold.

### Performance

| record threshold | open | ordered fetch | strided sparse fetch | append publish |
| ---: | ---: | ---: | ---: | ---: |
| 4K | 548.88 us | 6.540 ms | 2.427 ms | 192.77 ms |
| 16K | 210.98 us | 6.442 ms | 2.430 ms | 74.08 ms |
| 32K | 156.82 us | 6.585 ms | 2.429 ms | 59.98 ms |
| 64K | 130.90 us | 6.767 ms | 2.473 ms | 55.03 ms |
| 128K | 117.57 us | 6.806 ms | 2.480 ms | 51.40 ms |

Relative to the 4K default, 16K reduces open and append time by about 62% without changing strided sparse reads. At 32K, open and append improve by about 71% and 69%, while ordered and strided sparse reads remain within 1%. Moving from 32K to 64K yields another 16.5% open improvement and 8.3% append improvement, while ordered and sparse point estimates become 3.5% and 1.9% slower than 4K. Larger segments continue improving publication, but with diminishing returns and linearly increasing forced-normalization work.

### Normalization Amplification

A one-record insertion in the middle of an existing main segment was committed with patch publication disabled, forcing normalization:

| record threshold | input records | output records | amplification |
| ---: | ---: | ---: | ---: |
| 4K | 1 | 4,097 | 4,097x |
| 16K | 1 | 16,385 | 16,385x |
| 32K | 1 | 32,769 | 32,769x |
| 64K | 1 | 65,537 | 65,537x |
| 128K | 1 | 131,073 | 131,073x |

Larger segments make append publication and open cheaper because fewer immutable files are written and reopened, but foreground normalization rewrites the entire affected segment. The patch tier normally defers this cost; the probe deliberately disables it to expose the upper-bound tradeoff.

## K/V Geometry Sweep

All synthetic keys are ordered and have a stable prefix followed by an 8-byte sequence. This matches the parameter-key workload and exercises segment-level prefix stripping; notably, the 512-byte key profile has a 504-byte common prefix and should not be interpreted as a low-prefix or random-key result.

| profile | records | logical K/V bytes | purpose |
| --- | ---: | ---: | --- |
| 16-byte key / 16-byte value | 131,072 | 4 MiB | tiny fixed-shape result |
| 512-byte key / 64-byte value | 65,536 | 36 MiB | long, highly prefix-compressible key |
| 128-byte key / 1 KiB value | 32,768 | 36 MiB | medium result |
| 128-byte key / 16 KiB value | 2,048 | 32.25 MiB | large result |

The policies compare the current 4K-record/8-MiB default, a 16K/16-MiB intermediate policy, a 32K/16-MiB hybrid candidate, and 32K/32-MiB. The hybrid candidate raises the record cap for small values while retaining a 16 MiB normalization bound for profiles where the byte cap binds.

### Physical Layout

Each cell is `segment files / average physical segment size`.

| profile | 4K / 8 MiB | 16K / 16 MiB | 32K / 16 MiB | 32K / 32 MiB |
| --- | ---: | ---: | ---: | ---: |
| key 16 / value 16 | 32 / 88 KiB | 8 / 353 KiB | 4 / 706 KiB | 4 / 706 KiB |
| key 512 / value 64 | 16 / 281 KiB | 4 / 1.10 MiB | 3 / 1.46 MiB | 2 / 2.19 MiB |
| key 128 / value 1 KiB | 8 / 4.03 MiB | 3 / 10.75 MiB | 3 / 10.75 MiB | 2 / 16.13 MiB |
| key 128 / value 16 KiB | 5 / 6.42 MiB | 3 / 10.70 MiB | 3 / 10.70 MiB | 2 / 16.05 MiB |

The current record cap produces very small physical segments for tiny values. The 32K/16-MiB candidate reduces the 16/16 profile from 32 files to 4. For medium and large values, the 16 MiB byte cap controls the layout, so increasing the record cap has no additional effect.

### Forced Normalization

The table reports logical bytes republished after inserting one record into the middle of a main segment with patch publication disabled.

| profile | 4K / 8 MiB | 16K / 16 MiB | 32K / 16 MiB | 32K / 32 MiB |
| --- | ---: | ---: | ---: | ---: |
| key 16 / value 16 | 128 KiB | 512 KiB | 1.00 MiB | 1.00 MiB |
| key 512 / value 64 | 2.25 MiB | 9.00 MiB | 16.00 MiB | 18.00 MiB |
| key 128 / value 1 KiB | 4.50 MiB | 16.00 MiB | 16.00 MiB | 32.00 MiB |
| key 128 / value 16 KiB | 8.02 MiB | 16.01 MiB | 16.01 MiB | 32.01 MiB |

The 32K/32-MiB policy doubles the worst-case normalization range for medium and large values. The 32K/16-MiB policy preserves the 16 MiB bound while still fixing record-driven fragmentation.

### Current Default Versus 32K / 16 MiB

| profile | open | append publish | ordered fetch | strided sparse fetch |
| --- | ---: | ---: | ---: | ---: |
| key 16 / value 16 | 457.82 us -> 88.02 us | 176.98 ms -> 38.52 ms | 5.213 ms -> 5.201 ms | 1.867 ms -> 1.839 ms |
| key 512 / value 64 | 369.97 us -> 203.86 us | 113.10 ms -> 52.20 ms | 4.202 ms -> 4.289 ms | 1.196 ms -> 1.205 ms |
| key 128 / value 1 KiB | 428.57 us -> 363.25 us | 153.06 ms -> 111.53 ms | 4.771 ms -> 4.855 ms | 3.532 ms -> 3.529 ms |
| key 128 / value 16 KiB | 365.79 us -> 340.36 us | 134.29 ms -> 118.54 ms | 3.358 ms -> 3.395 ms | 145.82 us -> 155.90 us |

Open and read measurements use a 10-sample diagnostic run; corrected append publication uses 20 samples. Read point estimates are generally within about 2%. The large-value sparse outlier is not accompanied by a layout difference from the 16K/16-MiB policy and is treated as measurement noise rather than a segment-policy effect. The 16 KiB-value 32K/16-MiB append estimate also has a wide confidence interval; because its physical layout equals 16K/16-MiB, the difference is noise rather than a policy effect.

## Conclusion

The 4K default is physically fragmented for the intended small-result workload. Across layout, open, corrected append, read, and forced-normalization measurements, 32K records and 16 MiB remains the best balanced candidate: the record cap removes tiny-value fragmentation, while the byte cap bounds medium and large-value normalization at approximately 16 MiB. A 32 MiB byte cap sometimes saves one file and improves append publication, but it offers no consistent read benefit and doubles forced-normalization work for medium and large values. This benchmark records evidence only; it does not change the library defaults.
