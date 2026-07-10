# Remote Blob Backend Design

## Status

This is a future design note. It is not implemented by the current local filesystem backend and should be treated as a catalog/backend revision, not as an incremental change to the v1 local `MANIFEST` format.

## Purpose

The segment cache store is a good fit for object storage because visible data already consists of immutable segments plus a small manifest snapshot. A remote backend should preserve that shape instead of trying to emulate a mutable local filesystem. Segments become immutable blob objects, manifests become immutable versioned snapshots, and one small mutable reference object points to the latest manifest.

The target use case is distributed computation caching: workers can publish batches at coarse granularity, readers can download and cache immutable segments locally, and later merge/normalization can converge duplicate key copies through the existing deterministic winner rule.

## Goals

- Use object storage as a shared durable blob layer for immutable segment bytes.
- Keep reads coarse grained and local-cache friendly.
- Support optimistic multi-writer publication through compare-and-swap on a small reference object.
- Support pinned historical reads by manifest id.
- Preserve corruption-as-miss semantics for segment data.
- Avoid relying on object listing for visibility or latest-version discovery.
- Keep provider-specific APIs behind a small backend abstraction.

## Non-Goals

- A general-purpose remote database.
- Low-latency single-key reads from a cold cache.
- Cross-object transactions beyond publish ordering plus conditional reference update.
- Emulating POSIX rename, directory listing, or advisory locks on object storage.
- Background LSM-style compaction as the default write path.

## Object Model

The remote backend should use three object classes:

- Segment objects are immutable blobs containing one segment file's bytes.
- Manifest objects are immutable snapshots describing visible segment references.
- Reference objects are small mutable pointers to manifest objects.

Recommended object layout:

```text
<root>/
  store/identity
  refs/current
  refs/tags/<name>
  refs/pins/<pin-id>
  manifests/<manifest-id>.manifest
  segments/<segment-key>.seg
```

`store/identity` is the remote equivalent of `STORE`: it defines the namespace metadata, key length, value layout, checksum kind, compression kind, and remote catalog format. It should be created with "put if absent" semantics and never mutated after creation.

`refs/current` is the only object whose value defines the latest visible version. Readers must not scan `manifests/` and pick the highest generation, because object listing can be expensive, incomplete under failures, or semantically weaker than point reads. The latest version is whatever `refs/current` names.

## Latest Pointer

The current pointer should be a compact binary or JSON object with at least:

```text
RemoteRef {
  format_version
  store_identity_hash
  manifest_id
  manifest_hash
  generation
}
```

The storage provider's conditional-write token for `refs/current` is separate from this payload. On S3-like systems it may be an ETag or version id; on generation-based systems it may be object generation/metageneration. Internally call this token `ref_token`.

Reading the latest version is:

1. Read `refs/current` and capture both payload and `ref_token`.
2. Read `manifests/<manifest_id>.manifest`.
3. Verify `manifest_hash`, `store_identity_hash`, and manifest generation.
4. Build a runtime snapshot from the manifest's segment references.

Pinned reads skip step 1 and read a known manifest id directly. This is useful for long jobs, debugging, reproduction, and safe garbage-collection leases.

## Manifest Snapshots

Manifest snapshots should be immutable and self-describing. A remote manifest needs more identity than the current local v1 manifest because blob names should not depend on one process-local monotone `segment_id`.

Suggested manifest fields:

```text
RemoteManifest {
  format_version
  store_identity_hash
  manifest_id
  parent_manifest_id
  generation
  created_by
  entries[]
}

RemoteSegmentEntry {
  tier
  object_key
  content_hash
  segment_len
  segment_hash
  min_key
  max_key
}
```

`generation` is a human- and machine-friendly sequence number for a single reference history. It is not the authoritative latest selector by itself; `refs/current` is. `parent_manifest_id` records the optimistic update base and gives GC and debugging a version chain.

`manifest_id` should be content-derived when practical, for example a cryptographic hash over canonical manifest bytes. A UUID plus manifest hash also works, but content-addressed ids make retries and deduplication simpler.

## Segment Blob Identity

Remote segment object keys should be globally unique without requiring a central segment id allocator. Two reasonable strategies are:

- Content-addressed: `segments/<hash>.seg`, where `hash` is computed over full segment bytes.
- Writer-unique: `segments/<writer-id>/<uuid>.seg`, with `content_hash` stored in the manifest entry.

Content-addressed keys are preferred because upload retries are idempotent, local caches can key by content, and independently produced byte-identical segments deduplicate naturally. The existing non-cryptographic `segment_hash` is still useful for fast accidental replacement detection, but remote object identity should use a stronger content hash when blobs cross trust or failure boundaries.

The local `segment_id` concept can remain a logical ordering/debug field if needed, but it should not allocate remote object names. Remote writers should not coordinate only to obtain object names.

## Publish Protocol

Remote publication is "upload immutable objects, then CAS the current ref".

```text
read refs/current -> old_ref, old_ref_token
read old manifest
build candidate manifest from old manifest plus local changes
upload all new segment objects with put-if-absent semantics
upload immutable candidate manifest with put-if-absent semantics
conditional-put refs/current from old_ref_token to candidate ref
```

If the conditional reference update succeeds, the candidate manifest is the new latest version. If it fails, another writer won the race. The failed writer must read the new latest manifest, merge its still-unpublished logical changes with that latest snapshot, and retry. The already uploaded segment and manifest objects are harmless orphans until referenced or collected.

The important ordering invariant is that every segment object referenced by a manifest must be uploaded and verified before the manifest can become reachable from `refs/current`.

## Versioning Model

There are two distinct kinds of version:

- Provider object versions are storage-provider metadata for `refs/current` and possibly other objects. They are used as CAS tokens but should not be part of the portable catalog format.
- Store manifest generations are logical cache versions. They are stored in remote ref and manifest payloads and increment on successful publication.

Historical versions are simply old manifest objects. A snapshot is stable if the caller knows its `manifest_id`; it does not depend on the mutable current ref. Optional named references can make selected historical manifests discoverable:

```text
refs/tags/<name>      # human-named immutable or mutable tag
refs/pins/<pin-id>   # temporary GC pin for a long reader/job
refs/workers/<id>    # optional per-worker publication head
```

`refs/current` answers "what is latest now?" A manifest id answers "what exact snapshot do I want?" This split is the main reason not to overwrite a single `MANIFEST` object in place.

## Compare-And-Swap Requirements

Safe multi-writer publication requires a conditional update primitive on `refs/current`:

```text
compare_and_set_ref(name, expected_ref_token, new_ref_payload) -> Published | Conflict
```

The expected token must come from the exact ref version used to build the candidate manifest. If the backend cannot provide this primitive, it must not claim optimistic multi-writer safety. Valid alternatives are:

- A single writer per remote root.
- An external coordinator or lock service.
- A database-backed reference store for refs while object storage holds only immutable blobs.

Trying to discover conflicts by reading the current generation and then unconditionally overwriting `refs/current` is not sufficient; two writers can both read generation `N` and both publish `N + 1`, losing one update.

## Read Path

The remote read path should build a local runtime snapshot from a manifest, then use a local segment cache:

1. Resolve latest or pinned manifest.
2. Validate the manifest and store identity.
3. For each needed segment, check the local cache by `content_hash` or object key plus immutable version.
4. If absent, download the full segment object or a range containing the required block.
5. Verify segment fingerprint/content hash and ordinary block checksums before returning values.

The first implementation should prefer whole-segment downloads because it matches the current file-oriented reader and makes local cache behavior simple. Range reads can be added later for cold sparse lookups, but the segment/view layer should still treat the downloaded bytes as immutable backing data.

## Local Cache

The local cache is an optimization, not a correctness boundary. Cache entries should be keyed by immutable identity:

```text
LocalSegmentCacheKey {
  content_hash
  segment_len
}
```

When reading from cache, verify size and either the strong content hash or the existing segment fingerprint before accepting bytes. A corrupt or missing cache entry is a cache miss and should trigger a re-download or become a store miss according to the ordinary corruption policy.

Because segment objects are immutable, cached entries do not need invalidation when `refs/current` advances. Only reachability changes; object contents do not.

## Merge And Normalization

Remote writes should avoid turning every small local cache fill into a remote segment. Small writes can accumulate locally and flush as larger remote segments. Patch-tier segments are acceptable remotely as long as their count is bounded.

Normalization should be explicit or threshold-driven, not continuous background compaction. A remote normalizer reads a manifest snapshot, writes replacement segment objects for the normalized ranges, writes a candidate manifest, and publishes it through the same CAS protocol. If the CAS fails, the normalizer discards or retries from the new latest manifest.

This keeps object storage usage closer to a snapshot store than a high-churn LSM. It trades some temporary read amplification for fewer remote rewrites and lower coordination pressure.

## Garbage Collection

Garbage collection must be conservative because old readers may hold manifest ids that are no longer current.

Recommended policy:

- Treat every manifest reachable from `refs/current`, `refs/tags/*`, and `refs/pins/*` as live.
- Retain a configurable history window of parent manifests by generation or age.
- Mark every segment object referenced by retained manifests as live.
- Delete unreferenced manifest and segment objects only after a grace period.
- Never use directory/listing contents to decide visibility; listing is only a sweep mechanism.

Failed publications create two common orphan types: segment objects uploaded before a failed CAS, and manifest objects uploaded before a failed CAS. Both are invisible unless a later successful manifest references them.

Long-running jobs that need stable remote snapshots should create a temporary pin ref and delete it when done. If they do not pin, they must tolerate that very old manifest ids may eventually become unavailable after GC.

## Failure Semantics

| Failure | Visibility Result | Recovery |
| --- | --- | --- |
| Segment upload fails before manifest upload | No new data is visible | Retry upload or recompute |
| Segment upload succeeds but manifest upload fails | Uploaded segment is orphaned | Retry or later GC |
| Manifest upload succeeds but ref CAS fails | Manifest is orphaned | Merge with latest and retry |
| Ref CAS succeeds | New manifest is latest | Readers can discover it through `refs/current` |
| Referenced segment is missing or corrupt | That segment is miss space | Drop dead entry on next successful publication |
| Local cache entry is corrupt | Local cache miss | Redownload or treat as cache miss |

The remote backend should never return wrong data to hide these failures. If a remote object cannot be authenticated against the manifest reference, it is absent or corrupt, not a valid cache hit.

## Backend Abstraction Sketch

The storage abstraction should separate mutable refs from immutable objects:

```rust
trait RemoteBlobStore {
    fn get_ref(&self, name: &str) -> Result<RemoteRefRead>;
    fn compare_and_set_ref(
        &self,
        name: &str,
        expected: &RefToken,
        payload: &[u8],
    ) -> Result<RefUpdate>;

    fn get_object(&self, key: &str, range: Option<ByteRange>) -> Result<Bytes>;
    fn put_object_if_absent(&self, key: &str, bytes: &[u8]) -> Result<PutObjectResult>;
}
```

Provider-specific ETags, generation numbers, retries, multipart uploads, and authentication should live behind this trait. The store logic should only see immutable object keys, verified bytes, ref payloads, and conflict results.

## Open Questions

- Which strong content hash should remote segment and manifest ids use.
- Whether range reads are worth implementing before whole-segment caching is benchmarked.
- Whether remote manifests should preserve the local main/patch tier shape exactly or introduce a remote-specific tier optimized for worker imports.
- How long GC should retain unpinned historical manifests by default.
- Whether named branch refs are useful, or whether `refs/current` plus temporary pins are enough.
- Whether remote publication should be implemented as a new backend under the same public `Store` API or as an explicit `RemoteStore` with synchronization APIs.

## Summary

The remote backend should be modeled as an object-storage snapshot system: immutable segments, immutable manifests, and a CAS-updated latest pointer. `refs/current` tracks the latest version; manifest ids provide stable historical versions; optional pin/tag refs make selected versions discoverable and GC-safe. This design fits object storage because it avoids in-place mutation, amortizes remote IO over segment-sized blobs, and lets local caches reuse immutable bytes across readers and workflow phases.
