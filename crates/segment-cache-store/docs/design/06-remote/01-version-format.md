# Remote Version Format

## Scope

This document defines the future provider-neutral remote-v1 bytes. The format is not implemented and is independent of local `STORE` and `MANIFEST` v1. It reflects the pre-release decision to support both worker checkpoints and aggregate snapshots. The [remote repository design](00-design.md) defines behavior; the [implementation plan](02-implementation-plan.md) defines code boundaries.

## Store Identity

`RemoteStoreIdentityV1` identifies one logical cache namespace:

```text
RemoteStoreIdentityV1 {
  format_version: u8
  key_len: u32
  value_len: u32
  caller_metadata_len: u32
  caller_metadata: [u8; caller_metadata_len]
  structural_crc32c: u32
}
```

`value_len = 0` denotes variable-length values; a nonzero value denotes fixed-width values. `caller_metadata` is the same opaque compatibility value used by the local store; the caller encodes schema and semantic namespace versions in it. Platform and build fingerprints are not identity fields by default. Parsers enforce local geometry and bounded-length limits before allocation.

Compression and other writer tuning do not belong to remote identity. Each segment header self-describes its physical codec; supported codecs may coexist in one manifest, and a binary without a referenced codec reports an unsupported-segment error. The block checksum is the fixed local format constant. `AggregateOptions` selects physical policy only for newly encoded aggregate ranges; checkpoint and restore preserve verified segment bytes.

## Segment References

Every referenced object is one local-format segment file identified by its complete domain-separated `SegmentContentId`.

```text
RemoteSegmentRefV1 {
  content_id: [u8; 32]
  segment_len: u64
  min_key: [u8; key_len]
  max_key: [u8; key_len]
}
```

The declared range must equal the verified segment range. The verified header must match logical identity geometry, the fixed checksum ID, and a codec supported by the current operation.

`AggregateSnapshot` stores a counted, sorted `main_segments` array of these references. Ranges are strictly non-overlapping, content IDs are unique, and the total count and encoded bound bytes are bounded.

`WorkerCheckpoint` preserves local layout with explicit components:

```text
WorkerCheckpointV1 {
  main_count: u32
  main_segments[main_count]: RemoteSegmentRefV1
  patch_count: u32
  patches[patch_count]:
    owner_main_index: u32
    segment: RemoteSegmentRefV1
}
```

Main segments are sorted and strictly non-overlapping. Patch references are canonically sorted by `(owner_main_index, min_key, content_id)`. Each Patch names an existing Main index and its range is contained by that Main range. Each Main-owned PatchSet is strictly non-overlapping. Fixed parser limits and the repository's agreed local-layout policy cap total references, Patch files per Main, Patch bytes per Main, and encoded bound bytes before allocation; a manifest cannot choose or weaken those limits. No parser derives tier or ownership from object paths or ref names.

The canonical parser rejects duplicate segment references and duplicate content IDs. A trusted builder may collapse an exact duplicate before encoding, but canonical bytes never depend on parser-side deduplication.

## Manifest Snapshot

`RemoteManifestV1` is one immutable content snapshot with an explicit discriminant:

```text
RemoteManifestV1 {
  format_version: u8
  store_identity_id: [u8; 32]
  snapshot_kind: u8 // WorkerCheckpoint = 1, AggregateSnapshot = 2
  payload: kind-specific canonical fields
  structural_crc32c: u32
}
```

For `WorkerCheckpoint`, the payload is `WorkerCheckpointV1`. For `AggregateSnapshot`, it encodes only a counted sorted Main reference array. Unknown kinds are errors. Variant-specific structural validation occurs while parsing and exact length, codec, range, and byte-budget validation occurs again after segment-object verification.

`ManifestId` is the BLAKE3 ID of complete canonical manifest bytes. Empty aggregate snapshots are valid. A checkpoint's layout, including its valid Main-plus-Patch structure, is content-addressed; it contains no local IDs, provider tokens, or paths.

## Mutable Refs

```text
RemoteRefV1 {
  format_version: u8
  store_identity_id: [u8; 32]
  revision: u64
  manifest_id: [u8; 32]
  structural_crc32c: u32
}
```

`refs/current` must name an `AggregateSnapshot`, and `refs/workers/<worker-name>` must name a `WorkerCheckpoint`. The manifest discriminant controls decoding; the ref class supplies an additional expected-kind check rather than an implicit decoder choice. A complete manifest ID may address either kind directly. Provider compare-and-swap tokens are not portable bytes.

The first publication of a ref uses revision `1`; each replacement uses the checked successor of the previous revision. The number supports diagnostics and rejects malformed logical transitions, but the provider token returned by the preceding ref read remains the compare-and-swap authority.

Worker names use a restricted ASCII grammar and bounded length. Paths are derived from validated names and complete lowercase hexadecimal IDs.

## Object Layout And IDs

```text
<root>/
  store/identity
  refs/current
  refs/workers/<worker-name>
  manifests/<manifest-id>.manifest
  segments/<segment-content-id>.seg
```

Manifest and segment paths are derived from complete lowercase hexadecimal IDs and are not duplicated inside object payloads. V1 accepts complete IDs only.

```text
StoreIdentityId    = BLAKE3("scs-store-identity-v1\0" || canonical_identity_bytes)
SegmentContentId   = BLAKE3("scs-segment-v1\0" || complete_segment_bytes)
ManifestId         = BLAKE3("scs-manifest-v1\0" || canonical_manifest_bytes)
```

CRC32C is a structural check for bounded catalog payloads. Content IDs identify complete immutable bytes and are recomputed at the remote trust boundary.

## Encoding And Validation

Every format has explicit magic, one version, little-endian integers, fixed field order, bounded lengths, complete input consumption, and trailing CRC32C. Wire widths are not allocation budgets. Unknown versions, kinds, and codec IDs are hard errors.

Resolving a ref parses and validates the ref, verifies root identity, fetches and rehashes the named manifest, validates its kind-specific structure, then streams every referenced segment to verify length, content ID, structure, self-described codec support, and exact range. Bytes at a content-ID path that fail rehashing are corrupt.

Validation order is ref parse and root-identity comparison; manifest fetch, rehash, parse, and identity comparison; then streamed segment length, content-ID, structure, codec, and range verification. Ordinary local reads retain their existing lazy validation only after a verified snapshot has been locally published.

The implementation defines explicit limits for key and fixed-value lengths, caller metadata bytes, worker-name bytes, refs, manifests, total references per manifest, Patch files and bytes per Main, total encoded key-bound bytes, individual segment bytes, and streamed temporary files. Checked arithmetic enforces those limits before reservation, allocation, decompression, or transfer.

## Deferred Formats

V1 has no parent relationships, history, provenance objects, tags, pin leases, listing, retention policy, GC plan, or abbreviated-ID index.
