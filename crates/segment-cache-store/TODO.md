# segment-cache-store TODO

Implementation deltas, validation work, and genuinely open choices. The documents under [docs/design/](docs/design/00-overview.md) define the agreed target; [README.md](README.md) and public Rust documentation describe the current implementation. Items leave this file when the code and its independent contract tests land.

## 1. Main-Owned PatchSet Model (P0, designed)

Replace the current interval-connected, one-Patch runtime with the component model in [01-introduction.md](docs/design/01-introduction.md): one bounded Main owns a sorted, nonoverlapping PatchSet, and point depth remains Main plus at most one Patch.

- [ ] Choose concrete hard Main size, Patch files-per-Main, and Patch bytes-per-Main limits. These are fixed safety and amplification bounds, not per-manifest self-declared values.
- [ ] Encode the grouped target MANIFEST, with each Main followed by its counted Patch references, and construct a validated grouped runtime representation.
- [ ] Partition commits by Main ownership, gaps, and retained Patch barriers. Gap runs create Main output; in-Main gaps create Patch output; only intersected Patches are rewritten.
- [ ] Split new Patch and merge output before retained Patch or retained Main barriers so no output range crosses an unaffected segment.
- [ ] When a PatchSet would exceed either budget, synchronously normalize only that Main and PatchSet into bounded Main segments. Do not add background compaction or cross-component convergence.
- [ ] Keep sparse one-record gap commits as an explicit non-target rather than silently rewriting adjacent Main segments to improve occupancy.

Required tests cover multiple disjoint Patches under one Main, repeated inserts into the same Patch range, an insert between retained Patches, a write spanning several Patch ranges, exact budget boundaries, budget-triggered one-component normalization, a destination Main in a merge-source gap, merge output spanning retained Patches, and point/scan equivalence at Patch counts 1, 2, 4, and 8.

## 2. Destination-Biased Duplicate Semantics (P0, designed)

Replace storage-level value conflicts with deterministic first-healthy-winner semantics. A healthy visible destination key makes a commit record a no-op without comparing value bytes. An abnormal healthy Main/Patch duplicate resolves to Main on ordinary paths and is reported as `DuplicateVisibleKey` only by explicit verification.

- [ ] Remove `InputError::KeyConflict` from normal commit and local merge paths while retaining duplicate-key rejection inside one `WriteBatch`.
- [ ] Make local merge a left-biased union: destination first, then sources in explicit stable order. Later values for an existing healthy key are skipped without comparison.
- [ ] Ensure a corrupt or absent higher-priority value does not win; the next healthy source may supply the key only after its complete payload verifies.
- [ ] Add integration tests for differing bytes under one key, platform-sized floating-point differences, fixed source precedence, idempotence, associativity under fixed precedence, noncommutativity, and abnormal Main/Patch duplicates.

Scientific equivalence and numerical tolerance remain caller concerns. Semantic model or serialization changes require a new caller-metadata or key-namespace version; there is no per-key invalidation or repair API.

## 3. Ordinary Corruption-As-Absence (P0, designed)

Current fetch paths degrade classified block corruption to a miss, but commit filtering, normalization, and merge still use strict cursors and can wedge a damaged range. Align every ordinary local cache path with [05-corruption.md](docs/design/05-corruption.md); reserve strict behavior for explicit verify and trust-boundary operations.

- [ ] Make commit idempotence filtering, normalization, merge destination, and merge sources skip classified missing, format-failed, checksum-failed, or decompression-failed records at block granularity.
- [ ] Keep ordinary I/O, `STORE`, and `MANIFEST` failures hard.
- [ ] Require complete value-payload verification before an existing key or higher-priority merge source wins.
- [ ] Preserve healthy Main data while a corrupt block is refilled through Patch, then let foreground normalization remove the damaged representation. Do not add `BlockRepair`, tombstones, or per-key repair metadata.
- [ ] Add end-to-end corruption chains for metadata checksum, payload checksum, decompression, missing segment, malformed segment index, refill, normalization, and local merge. Explicit verify, import, export, checkpoint, and remote aggregation must fail on the same damaged inputs.

## 4. Pre-Release Format Deltas (P0)

The exact target is in [03-format.md](docs/design/03-format.md). Experimental stores are rebuilt rather than migrated.

- [ ] Make `STORE` persist only opaque caller metadata; move `key_len` and `value_len` ownership to MANIFEST.
- [ ] Replace the flat Main-then-Patch MANIFEST with grouped Main-owned PatchSets and validate format-wide Main/PatchSet limits before allocation.
- [ ] Let `CommitOptions` select compression for newly written segments and allow one manifest to reference self-describing segments with different supported compression IDs.
- [ ] Pin grouped MANIFEST canonical bytes and its XXH3-128 trailer, segment-header bytes, the segment Footer XXH3-128 trailer and prefix-compressed ranges, XXH3-128 known-answer vectors, raw/compressed frame derivation, and parser safety limits with golden tests.

## 5. Remote Checkpoint And Aggregate (future, designed)

Implement [06-remote/](docs/design/06-remote/00-design.md) only after the local component and corruption contracts are stable.

- [ ] Add provider-neutral identity, typed content IDs, explicit `SnapshotKind`, grouped `WorkerCheckpoint`, Main-only `AggregateSnapshot`, and kind-constrained refs.
- [ ] Add public local snapshot capture that pins segment handles against commit and garbage collection for the transfer lifetime.
- [ ] Checkpoint Main plus valid PatchSets by uploading exact verified self-describing segment bytes without mutating or normalizing the source; unsupported formats or codecs fail. Aggregate re-encodes only logically affected ranges.
- [ ] Stream and strictly verify every remote object through a bounded temporary artifact before local publication; upload immutable segments independently, then the manifest, then optionally compare-and-swap the ref.
- [ ] Make aggregate capture `refs/current` internally, then apply caller-specified or canonical lexicographic order to an explicitly supplied duplicate-free worker set. Use first-source-winner semantics and Main-only output; a compare-and-swap conflict recaptures current and replans.
- [ ] Keep the direct HPC baseline as one independent durable network-filesystem Store per worker. Scratch is optional and safe only with explicit checkpoint completion before loss.

## 6. Optional Source-Segment Adoption

A source Main segment disjoint from all destination ranges and untouched by source Patch data is eligible for byte-level adoption instead of transcoding: reflink where available, streamed copy otherwise, with a fresh local segment ID and the existing content ID and range.

- [ ] Decide whether local adoption always crosses the strict complete-file verification boundary or may trust a separately managed local root. Remote adoption is always strict.
- [ ] Benchmark the gain before adding filesystem-specific complexity.

## 7. Hardening And Validation

- [ ] Decide whether deterministic verification failures should be cached for an immutable segment. Successful verification is cached today; transient I/O errors must never become cached corruption.
- [ ] Add a `STORE` read-size limit and classify non-UTF-8 STORE bytes as `CatalogError::StoreParse`, not generic I/O.
- [ ] Make `Store::range(start, end)` and `visit_range` reject `start > end` before segment I/O, and fuse `RangeCursor` after its first error.
- [ ] Preserve explicit garbage collection semantics across platforms: it may make an older manifest snapshot no longer openable, but commit and open never collect automatically. Checkpoint capture requires in-process segment pins.
- [ ] Remove the duplicate `#[cfg]` in `frame.rs`, avoid writing a segment header before the empty-input check in `writer.rs`, replace the catch-all `_ => continue` in `open.rs`, and use the workspace `redb` dev-dependency.
- [ ] Add concurrency stress for commit versus existing readers, commit versus fresh read-only open, explicit garbage collection versus openers, and captured checkpoint pins.
- [ ] Add property or model-based coverage for batch routing across Main gaps, Patch barriers, dead segments, corruption, budget thresholds, and several components.
- [ ] Extend parameter-evolution benchmarks with repeated middle inserts, Patch counts 1, 2, 4, and 8, writes touching one versus several Patches, PatchSet budget normalization, and the intentionally unsupported sparse one-record gap case.
