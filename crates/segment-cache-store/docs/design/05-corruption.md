# Corruption Semantics And Trust Boundary

## Ordinary Cache Paths

Classified missing data, format failure, checksum failure, and decompression failure are logically absent in ordinary `fetch`, `contains`, commit idempotence filtering, normalization, and local cache merge. A value is an existing destination winner only after its full payload verifies. This lets a recomputation refill a corrupt location through the ordinary PatchSet path and later foreground normalization; the design has no `BlockRepair` metadata or tombstones.

The classified scope is explicit:

- A missing referenced segment, invalid segment length, bad header or footer CRC, incompatible schema or codec, malformed sparse index, or segment-range disagreement makes the complete segment absent.
- A bad block lookup digest, malformed block geometry, disagreement with the validated block-index range, bad value-frame digest, or decompression failure makes only that block absent.
- An abandoned temporary file or a final segment not referenced by MANIFEST is invisible rather than data.

Ordinary I/O failures are hard errors. `STORE` and `MANIFEST` failures are also hard errors because they define namespace identity and visible membership. Final segment files not referenced by MANIFEST and abandoned `.tmp` files are invisible; explicit garbage collection may later remove them.

## Strict Boundaries

Explicit verify, import, export, checkpoint, and remote aggregation are strict. They reject classified corruption and report `DuplicateVisibleKey` for an unexpected logical duplicate. They do not silently omit damaged data. A remote or externally materialized segment is fully verified before use, including complete-file identity and all block checks.

## Duplicate Handling Under Damage

Normal publication does not introduce duplicates. If an abnormal Main/Patch duplicate is present, ordinary read and normalization choose a healthy Main. If Main is absent or corrupt, a healthy Patch may supply the record. Explicit verification reports the duplicate even when the ordinary path has a deterministic winner.

## Integrity Layers

| Bytes | Integrity check | Role |
| --- | --- | --- |
| Catalog and fixed structure: MANIFEST, header, footer | CRC32C | Detect structural corruption |
| Block lookup metadata and value payload | Fixed unseeded XXH3-128 | Verify lookup routing and returned payload bytes |
| Complete segment crossing a trust boundary | BLAKE3 `SegmentContentId` | Cryptographic content identity |

Locally published immutable files are trusted on ordinary open after file length, header, footer, schema, range, and offset validation. Ordinary open does not rescan the complete file to compare `SegmentContentId`; externally replacing published local files is outside the managed-root contract. Block verification is lazy: an accessed block verifies its lookup metadata, and its payload verifies before a hit returns. Process-local reuse may avoid repeating a successful verification. This assumes a non-adversarial managed local filesystem; checksums are not authentication for an attacker who can replace bytes and digests.

Externally materialized bytes are untrusted. Import, restore, checkpoint ingestion, and remote aggregation stream the complete segment through declared-length and `SegmentContentId` verification, validate all structural fields and exact ranges, and verify every block and compression frame before local publication.

`SegmentContentId` is mandatory for trust-boundary crossings but is not itself an authentication policy. The repository or caller must authenticate remote manifests and authority separately.

## Fixed Mandatory Block Checksum

Every block carries unseeded XXH3-128 over its lookup metadata and over its value-payload frame. Format ID `1` fixes both the algorithm and its canonical big-endian 16-byte representation. It is not a create-time option and has no `None` kind. An unknown ID is unsupported, so an algorithm change needs a new format ID rather than silently changing persisted semantics. Known-answer tests over empty, short, and long inputs pin cross-platform output and prevent a dependency update from silently changing persistent bytes.

The fixed checksum is mandatory because a checksum-less cache cannot later establish that an exported or merged input was intact. Split block digests amortize the cost: metadata verifies when first needed, payload verifies only for a returned value or scan, and successful checks reuse process-local state. CRC32C remains appropriate for short fixed structures where its algebraic guarantees are useful; for large random block payloads, XXH3-128 provides a much wider probabilistic floor than a 32-bit CRC at only 16 additional bytes per block pair compared with two 64-bit digests. `SegmentContentId` supplies the cryptographic check required when complete bytes cross a trust boundary.

Checksum verification may be disabled only on a read-only handle for benchmarking. That explicitly weakens the cache-safe hit guarantee. Writable handles verify checksums so a foreground normalization or local merge cannot transcode unchecked bytes into new segments.

## Current Implementation Notes

The current implementation already treats read-path block corruption as a miss, but ordinary commit idempotence filtering, normalization, and merge use strict cursors and can be wedged by a corrupt block. It also retains `KeyConflict` behavior for differing bytes during publication and merge. The target corruption-as-absence behavior for all ordinary paths, PatchSet refill, and strict-only verification remains to be implemented and regression-tested.
