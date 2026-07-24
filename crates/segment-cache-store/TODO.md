# segment-cache-store TODO

Open design decisions and planned work. The design documents under [docs/design/](docs/design/00-overview.md) describe the agreed current design; everything not yet implemented or not yet decided is tracked here. Items leave this file as they land. Once this work is upstreamed, this file may be replaced by the repository issue tracker.

## 1. Merge Barrier-Aware Output Splitting (P0, designed)

v1 cuts merge output chunks by the record and byte thresholds only, so output straddling a retained destination segment fails loudly at manifest validation (`SegmentOverlap`, or `MultiplePatchesInComponent` across two retained patches). The failure is atomic and never corrupts the destination snapshot, but merge is unusable in that shape. The agreed fix is the third split trigger in [04-operations.md](docs/design/04-operations.md): barriers are the ranges of retained destination segments (main and patch, sorted by `(min_key, segment_id)`), and a non-empty chunk flushes before it would straddle a barrier.

- [ ] Implement barrier splitting in `StoreMergeRecords::push_next_into` (src/commit/merge.rs), with the barrier set built from the plan's retained segments after `retire_components_for_ranges`.
- [ ] Regression test: destination main segment sitting in a source gap (previously `SegmentOverlap`).
- [ ] Regression test: merge output spanning two retained patches (previously `MultiplePatchesInComponent`).
- [ ] Regression test: merge followed by explicit `Store::normalize`.

## 2. Source-Segment Adoption (future, decision open)

Source main-tier segments disjoint from all destination ranges and untouched by source patches are pure inserts, so they are eligible for byte-level adoption instead of transcoding: reflink copy where supported, streamed copy otherwise, with a fresh destination `segment_id`, the copied `content_id`, and the source entry's key range.

- [ ] Decide whether adopted bytes are trusted like locally published files or verified through the full single-segment path (`SegmentContentId` recomputation plus per-block checks), then implement.

## 3. Write-Path Corruption-As-Miss (P0, decision open)

Three administrative read sites run strict and abort the whole operation on a corrupted block: the idempotency/conflict check before publication, the normalization input stream, and the merge destination input stream. A key whose old value sits in a corrupted block reads as a miss but cannot be written back, and every commit overlapping the component fails: the range is wedged for writes. The only recovery is manual segment deletion, which forgets healthy records too. See [05-corruption.md](docs/design/05-corruption.md).

Proposal A: block-granular forgetting on administrative reads, a `corruption_detected` signal on commit stats, and healing through immediate or scheduled normalization. The merge **source** stream stays strict regardless. Documented trade-off: `KeyConflict` protection is forfeited for keys inside corrupted blocks.

- [ ] Decide between A with an opt-out flag and A with automatic healing normalization (A+B).
- [ ] Implement the chosen variant.
- [ ] Add a "block corruption -> write -> recover" integration test chain (current recovery tests cover only missing files).

## 4. Open Design Decisions

- [ ] Global patch/component budget: disjoint components may each retain one patch, so many small disjoint interleaving batches accumulate patches without bound (scan fan-in, open cost, manifest size). Decide whether this is an accepted workload assumption or needs a global threshold that forces normalization.
- [ ] Verification-failure caching: only successful verifications are recorded today, so a corrupted block is re-read and re-checked on every probe until its segment is retired. Deterministic failures (checksum, layout) can be cached safely under the immutability contract; transient IO errors must not be. Would add a failed state to `BlockVerification` in src/segment/state.rs.
- [ ] Pin the GC vs fresh-opener race semantics in the docs: a reader that loaded `MANIFEST` before a concurrent commit + GC observes misses (not errors) for collected segments, and deleting open files behaves differently on Windows than on Unix.

## 5. Format Target-Model Deltas

The v1 implementation notes in [03-format.md](docs/design/03-format.md) track these; rebuilding experimental stores is the migration strategy.

- [ ] `STORE` persists only caller metadata (drop `key_len`, `value_len`, `block_checksum_id`, `value_payload_compression_id`).
- [ ] `MANIFEST` owns `value_len`.
- [ ] `CommitOptions` selects the value-payload compression kind.
- [ ] Open allows mixed-compression segment sets.

## 6. Small Hardening And Contract Fixes

- [ ] `STORE` read size limit (mirroring the 256 MiB manifest bound), and report non-UTF-8 `STORE` as `CatalogError::StoreParse` instead of `Error::Io` so callers classifying failures by error type do not misjudge.
- [ ] Define `Store::range(start, end)` behavior for `start > end` (prefer an input error over a silently empty result).
- [ ] Fuse the `RangeCursor` iterator after an error so callers that ignore errors do not get an infinite error stream.
- [ ] Cleanups: duplicate `#[cfg]` in frame.rs, header written before the empty check in writer.rs, `_ => continue` in open.rs, dev-dep `redb` pinned via the workspace.
- [ ] Test gaps: concurrency stress (commit vs concurrent readers, GC vs readers — the process model promises this but nothing verifies it) and property/fuzz coverage of the write routing space (direct/patch/normalize x dead entry x multiple components).
