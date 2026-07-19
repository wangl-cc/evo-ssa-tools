# Remote Version Format

## Scope

This document defines the provider-neutral v1 identity, config, manifest, segment-reference, and ref formats for the future remote repository. The [remote repository design](remote-blob-design.md) defines workflows, and the [implementation plan](remote-implementation.md) defines code boundaries.

These formats are independent of local `STORE` and `MANIFEST` v1. Remote support does not migrate or reinterpret existing local catalog bytes.

## Logical Store Identity

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

`value_len=0` means variable-length values; any non-zero value means fixed-width values. The identity parser applies the same key and value geometry limits as the local store before allocating caller metadata.

Block checksum and value compression do not belong to logical identity. Stores with equal geometry and caller metadata may exchange records even when their local physical codecs differ.

`StoreIdentityId` is the BLAKE3 identity of the complete canonical identity bytes.

## Repository Config

One immutable `RemoteRepositoryConfigV1` selects the physical segment codecs for a v1 remote root:

```text
RemoteRepositoryConfigV1 {
  format_version: u8
  store_identity_id: [u8; 32]
  block_checksum_kind: u8
  value_payload_compression_kind: u8
  structural_crc32c: u32
}
```

Unknown codec IDs are rejected. A build that does not contain the selected codec reports an unsupported-config error rather than silently treating the object as absent.

The v1 config is immutable after repository creation. Its `RepositoryConfigId` covers the complete canonical config bytes. Config is separate from logical identity so a future explicit migration can rewrite physical objects while retaining the logical namespace.

## Segment References

Remote manifests contain concrete remote segment references:

```text
RemoteSegmentRefV1 {
  content_id: [u8; 32]
  segment_len: u64
  min_key: [u8; key_len]
  max_key: [u8; key_len]
}
```

The referenced object is exactly one existing local-format segment file. `content_id` is the existing domain-separated `SegmentContentId` over the complete segment bytes.

A validated segment set guarantees:

- every range has the identity's key length
- every range has `min_key <= max_key`
- references are sorted by `min_key`
- ranges are strictly non-overlapping
- content IDs are unique
- segment lengths and total encoded key-bound bytes stay within implementation limits
- every verified segment header matches the repository config
- every verified segment's actual range equals its manifest range

The canonical parser rejects duplicate references. A manifest builder may deduplicate exact references before encoding, but canonical bytes never rely on parser-side collapsing.

Range non-overlap is a catalog invariant. Logical key uniqueness is established by the strict snapshot builder that creates the manifest, not inferred solely from range metadata.

## Manifest Snapshot

`RemoteManifestV1` is one immutable content snapshot:

```text
RemoteManifestV1 {
  format_version: u8
  store_identity_id: [u8; 32]
  repository_config_id: [u8; 32]
  segment_count: u32
  segments: [RemoteSegmentRefV1; segment_count]
  structural_crc32c: u32
}
```

`ManifestId` is the BLAKE3 identity of the complete canonical manifest bytes. The manifest does not store its own ID.

The snapshot contains no timestamp, writer ID, parent list, local segment ID, patch tier, allocation counter, provider token, or object path. Those fields either make equivalent content unnecessarily distinct or belong to publication state rather than visible cache content.

An empty manifest is valid and represents an empty cache snapshot.

## Mutable Refs

`RemoteRefV1` is a bounded mutable pointer:

```text
RemoteRefV1 {
  format_version: u8
  store_identity_id: [u8; 32]
  repository_config_id: [u8; 32]
  revision: u64
  manifest_id: [u8; 32]
  structural_crc32c: u32
}
```

V1 defines two ref classes:

- `refs/current` selects the shared aggregate snapshot.
- `refs/workers/<worker-name>` selects one worker-owned snapshot.

`revision` increases by one for each successful update to one ref name. It aids diagnostics and rejects malformed logical transitions, but the provider's conditional-write token remains the authority for compare-and-swap. Provider tokens never appear in portable bytes.

Worker names use one restricted ASCII grammar and a bounded encoded length. Path construction is performed from a validated `WorkerName`; callers cannot supply arbitrary object paths.

## Object Layout

```text
<root>/
  store/identity
  store/config
  refs/current
  refs/workers/<worker-name>
  manifests/<manifest-id>.manifest
  segments/<segment-content-id>.seg
```

Manifest and segment paths are derived from complete lowercase hexadecimal IDs. IDs and derived paths are not duplicated inside object payloads.

V1 accepts only complete IDs. Abbreviated-ID resolution would require listing or an index and is deferred.

## Content IDs

All immutable IDs are domain-separated BLAKE3 values:

```text
StoreIdentityId    = BLAKE3("scs-store-identity-v1\0" || canonical_identity_bytes)
RepositoryConfigId = BLAKE3("scs-repository-config-v1\0" || canonical_config_bytes)
SegmentContentId   = BLAKE3("scs-segment-v1\0" || complete_segment_bytes)
ManifestId         = BLAKE3("scs-manifest-v1\0" || canonical_manifest_bytes)
```

The local implementation already computes and stores `SegmentContentId`; remote support reuses it rather than introducing another segment fingerprint.

Structural CRC32C and content IDs have different ownership. CRC32C gives each catalog payload a fixed local structural check. A content ID identifies complete immutable bytes and is recomputed when bytes cross the remote trust boundary.

## Canonical Encoding

Every remote catalog format uses:

- an explicit magic before the fields shown above
- one v1 format version
- little-endian integers
- fixed field order
- bounded lengths checked before allocation
- complete input consumption
- no ignored extension bytes
- canonical segment order
- a trailing CRC32C over all preceding bytes

Unknown versions and IDs are hard errors. Parsers validate external bytes once and construct types with private fields; internal repository code relies on those validated invariants.

## Validation Order

When resolving a ref, the repository validates in this order:

1. Parse and structurally validate the bounded ref bytes.
2. Compare identity and config IDs with the opened repository root.
3. Fetch the exact manifest object named by `manifest_id`.
4. Recompute `ManifestId`, then parse and validate the manifest.
5. Compare manifest identity and config IDs with the ref and root.
6. Materialize each required segment and recompute its length and `SegmentContentId`.
7. Validate segment structure, physical config, and exact key range before using it for strict synchronization.

Ordinary local reads retain their existing lazy block validation and corruption-as-miss behavior after a verified snapshot has been published locally.

## Parser Limits

The implementation defines explicit limits for:

- key and fixed-value lengths
- caller metadata bytes
- worker-name bytes
- segment count per manifest
- manifest bytes
- total encoded key-bound bytes
- individual segment bytes
- ref bytes

Wire integer widths are not allocation budgets. Limits are checked with checked arithmetic before reserving memory or starting an object transfer.

## Deferred Formats

History commits, parent relationships, operation provenance, tags, pin leases, retention policies, and GC plans are not part of v1. They may be added as separate objects without changing the content-snapshot manifest.
