# Corruption Semantics And Trust Boundary

How the store behaves when bytes go bad, and how much it trusts the bytes it reads. The first contract from [01 Introduction](01-introduction.md) governs everything here: values may be forgotten on crash or corruption because they are recomputable, but the store must never return silently corrupted data as a valid hit.

## Corruption-As-Miss

Missing, malformed, or corrupted data is treated as absent, never as valid output. Concretely:

- Corrupted segment header: whole segment ignored.
- Corrupted segment footer or block index: whole segment ignored.
- Manifest entry pointing to a missing segment: entry is invisible and later dropped.
- Manifest entry whose min/max range does not match the first and last indexed block ranges: segment ignored.
- Corrupted block metadata checksum: that block behaves like missing data.
- Corrupted block value payload frame checksum: matching keys in that block behave like missing data.
- Malformed block layout: that block behaves like missing data.
- Temp file after crash: ignored.
- Final segment not referenced by manifest: ignored and later collected.

Visibility is readability. A corrupted block is detected deterministically — checksum mismatch, layout violation, or disagreement with its index entry are pure functions of the immutable file bytes — so every read path misses on it forever. A key inside a corrupted block is effectively absent everywhere; forgetting it at block granularity never produces two visible values for one key.

## The Trust Boundary

The store applies one trust rule everywhere: **immutable files published under a managed local store root are trusted; bytes that cross a trust boundary are verified before use.** The rule assumes a non-adversarial local filesystem: checksums detect accidental corruption (bit rot, torn writes, misdirected IO), not an attacker who recomputes digests after modifying bytes, and `SegmentContentId` is a content identity, not an authentication tag.

The boundary has three layers, each with a check matched to its threat:

| Layer | Check | Strength source |
| --- | --- | --- |
| Catalog and fixed layout (header, footer, manifest) | CRC32C | Algebraic guarantees at short, fixed-layout sizes |
| Block lookup metadata and value payload (large, random corruption) | XXH3 128-bit | 2^-128 probabilistic floor, high throughput |
| Whole-segment content identity (trust boundary crossings) | BLAKE3 `SegmentContentId` | Cryptographic strength |

Concretely:

- Ordinary local open compares the file length and validates the header, footer CRC, schema, ranges, and offsets of each referenced segment. It does not rescan every segment to recompute its `SegmentContentId`, so a structurally valid, range-matching replacement of a published file is accepted. Externally replacing published local files is unsupported; latent corruption introduced that way is absorbed later as corruption-as-miss on the affected blocks.
- Externally materialized segment bytes — anything not published under the managed root, including future remote ingestion — go through the full single-segment verification path: `SegmentContentId` recomputation over the complete file plus per-block checksum and compression-frame validation.
- Checksum verification on reads can be disabled only for read-only opens. This is an explicit benchmarking mode, not the cache-safe default. Writable opens require checksum verification so normalization cannot merge corrupted bytes into freshly checksummed output.

## The Block Checksum Is Fixed And Mandatory

Every block in every store carries the same checksum: unseeded XXH3 128-bit over the lookup metadata and over the value payload frame (format id `1`, canonical 16-byte digests). The checksum is a store-wide constant, not a create-time choice; there is no `None` kind and no checksum feature flag. The header `block_checksum_id` byte is a fixed discriminant: any value other than `1` is unsupported and the segment is treated as absent, which gives a future algorithm change a clean fail-safe path.

Why this design:

- **Mandatory.** The store's first contract ("never return corrupted data as a valid hit") should not be forfeitable by a create-time option. A checksum-less namespace can never be verified later, and its corruption would propagate: merge and remote publication transcode source records into freshly checksummed output, so unverifiable source bytes would be laundered into verified segments. With the checksum mandatory, the merge source's forced strict verification is never vacuous and no special casing is needed anywhere.
- **Amortized on reads.** Split checksums skip payload verification for miss-only probes, and process-local verification reuse makes each block's verification happen at most once per process. Checksum throughput still affects warm-page first-touch performance, but it is not a recurring cost after that block has been verified.
- **XXH3-128 over CRC32C for blocks.** CRC's algebraic guarantees (burst errors, few-bit errors) are strongest on short messages and degrade as the payload grows; at block sizes of 16 KiB–128 MiB the practical corruption modes (torn writes, sector scribbles) exceed those guarantees and both checksum families fall back to their probabilistic floors, where a 128-bit hash (2^-128) dominates a 32-bit CRC (2^-32). CRC32C is kept for the short fixed-layout structures (header, footer, manifest) where its algebraic guarantees are fully effective.
- **128 bits over 64 bits.** A 64-bit checksum is already adequate for non-adversarial local corruption, but the format is intended to remain stable and participate in future remote workflows. Each block carries two digests, so the wider output costs 16 additional bytes per block: less than 0.1% at the default 16 KiB target block size. XXH3-128 has nearly the same hashing cost as XXH3-64, making the wider collision margin a proportionate format choice.
- **Pinned semantics.** Format id `1` means exactly "unseeded XXH3-128 serialized in its canonical big-endian 16-byte representation". Known-answer tests cover empty, short, and long inputs so a dependency upgrade cannot silently change persisted output; a future algorithm revision would receive a new format id, never a silent change.

## Write Path Behavior

Read paths degrade corrupted blocks to cache misses. The write path currently does the opposite: three administrative read sites run in strict mode and abort the whole operation when they touch a corrupted block — the idempotency/conflict check before publication, the normalization input stream, and the merge destination input stream.

The consequence for the cache workload: a key whose old value sits in a corrupted block reads as a miss, so the caller recomputes it — but the recomputed value cannot be written back, because the idempotency check probes the same corrupted block and fails the commit. Once a component contains a corrupted block, every commit overlapping that component routes to normalization and also fails: the key range is wedged for writes. The only recovery is manual deletion of the segment file so its manifest entry becomes dead, which forgets every record in the segment, healthy blocks included.

Relaxing the write path to the same corruption-as-miss rule — block-granular forgetting on administrative reads, a `corruption_detected` signal on commit stats, and healing through immediate or scheduled normalization — is a pending decision tracked in [TODO.md](../../TODO.md). Note that this relaxation also forfeits `KeyConflict` protection for keys inside corrupted blocks: a recomputed value that differs from the forgotten original would be accepted silently, which is consistent with treating corruption as absence. The merge **source** stream stays strict with forced checksum verification regardless: a corrupt source fails the merge loudly so the caller rebuilds the worker instead of silently aggregating half a store. Because the block checksum is mandatory, this strictness is always effective.
