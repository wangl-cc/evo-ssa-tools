# Remote Repository Implementation Plan

## Scope

This plan adds the minimum provider-neutral infrastructure required for explicit worker push, pull, and strict aggregation. It does not implement a provider SDK and does not change the public local `Store` API or local v1 formats.

The [remote repository design](remote-blob-design.md) defines behavior. The [remote version format](remote-version-format.md) defines persistent bytes and identities.

## Existing Assets

The current crate already provides the difficult local primitives:

- deterministic immutable segment writing
- an unconditional domain-separated BLAKE3 `SegmentContentId`
- complete segment verification at an external trust boundary
- validated header, footer, block, and key-range decoding
- strict cross-store merge with unique-key conflict semantics
- bounded main-segment output through the snapshot builder
- atomic local manifest publication

Remote implementation should reuse these concepts directly. It must not add another segment hash, another winner policy, or a generic catalog framework around the local store.

## Crate Boundary

The provider-neutral engine initially lives under a private `remote` module behind an optional `remote-repository` feature. Only the domain types and provider capabilities needed by adapter crates become public.

Provider SDKs, credentials, retry middleware, async runtimes, and transport configuration live in separate adapter crates. The first repository API is synchronous, matching `Store` and `scs`.

Do not refactor the local catalog into generic `CatalogHead<I>`, `MainSegmentSet<I>`, `CatalogTransition<I>`, or `SegmentMaterializer<I>` abstractions. There is no second implementation with equivalent local allocation and publication semantics. Add concrete remote types and extract shared code only where local merge and remote aggregation execute the same strict snapshot-building algorithm.

## Internal Modules

The initial module boundary is:

```text
remote/
  identity.rs     logical identity and repository config
  manifest.rs     concrete remote segment set and canonical manifest codec
  reference.rs    validated worker names, ref codec, and CAS transition rules
  object.rs       typed object IDs and derived object names
  materialize.rs  streamed download, verification, and local cache publication
  publish.rs      immutable-object ordering and ref CAS
  sync.rs         push, pull, and worker aggregation orchestration
  provider.rs     narrow provider capabilities
```

Keep codecs next to the validated types they construct. Fields remain private. Parsing validates external bytes once; downstream code receives trusted identity, config, manifest, ref, and object-name values.

## Concrete Domain Types

The remote domain needs concrete types rather than generic backend identity parameters:

```rust
struct RemoteSegmentRef {
    content_id: SegmentContentId,
    segment_len: u64,
    min_key: Vec<u8>,
    max_key: Vec<u8>,
}

struct RemoteManifest {
    store_identity_id: StoreIdentityId,
    repository_config_id: RepositoryConfigId,
    segments: Vec<RemoteSegmentRef>,
}
```

`RemoteManifest::parse` validates canonical order, unique content IDs, range geometry, count limits, and complete input consumption. `RemoteManifestBuilder` accepts verified output artifacts from the strict snapshot builder and emits canonical references. The parser rejects duplicates; only the builder may deduplicate an exact reference before encoding.

The segment writer already returns the facts publication needs through its existing result and manifest entry path. Do not add a speculative `SegmentWriteSummary` carrying an unused aggregate record count.

## Shared Snapshot Builder

Local normalization, `Store::merge_from`, worker push that requires re-encoding, remote aggregation, and pull into a differently configured local root all need the same operation:

```text
strict sorted input cursors
  -> reject malformed input and conflicting values
  -> collapse byte-identical records
  -> write bounded main segments with target options
  -> return staged verified artifacts
```

This is one real shared concept and should remain independent of local or remote publication. It accepts target segment options explicitly and produces staged files plus content IDs and key ranges. The caller decides whether those files receive local segment IDs or remote object names.

Read cursors used by this builder are strict. Corruption aborts synchronization; they do not inherit the ordinary cache-read policy that may degrade a corrupt block to a miss.

## Verified Artifact Boundary

Bytes entering from a remote provider are untrusted until one operation has:

1. streamed them into a temporary file with a maximum length
2. compared the actual length with the manifest reference
3. recomputed the complete `SegmentContentId`
4. validated segment header, footer, config, structure, and exact key range
5. synchronized and atomically published the file into the local content cache

An internal `VerifiedSegmentArtifact` may carry the resulting path and validated reference between verification and publication. It is short-lived and consumable, not a persistent trust marker and not a public token.

Ordinary local open continues to trust files already published under the local store contract. Remote verification does not add a full-file rescan to every open.

## Provider Capabilities

Segment transfer must be streaming. A provider adapter must not require `Vec<u8>` for a complete segment:

```rust
use std::io::{Read, Write};

pub trait ImmutableObjects {
    type Error: std::error::Error + Send + Sync + 'static;

    fn get_to(
        &self,
        id: &ObjectId,
        sink: &mut dyn Write,
    ) -> Result<GetResult, Self::Error>;

    fn put_if_absent_from(
        &self,
        id: &ObjectId,
        len: u64,
        source: &mut dyn Read,
    ) -> Result<PutResult, Self::Error>;
}

pub trait MutableRefs {
    type Error: std::error::Error + Send + Sync + 'static;

    fn get(&self, name: &RefName) -> Result<Option<RefRead>, Self::Error>;

    fn compare_and_set(
        &self,
        name: &RefName,
        expected: Option<&RefToken>,
        next: &[u8],
    ) -> Result<RefUpdate, Self::Error>;
}
```

`get_to` writes into a repository-owned bounded temporary file. `put_if_absent_from` reads from a verified local file. If an adapter retries after a partial transfer, it must request a newly opened source from repository orchestration rather than assuming a consumed reader can rewind.

Ref payloads are intentionally bounded and may be held in memory. Immutable insertion is idempotent, ref conflict is distinct from transport failure, and listing or deletion is not required by either v1 trait.

Do not introduce one `StorageBackend` trait containing local locks, rename, directory sync, object streaming, listing, compare-and-swap, and deletion. Those operations do not share semantics.

## Public API

The first public API exposes synchronization intent rather than wire structs or object paths:

```rust
pub struct StoreIdentityId(...);
pub struct RepositoryConfigId(...);
pub struct ManifestId(...);
pub struct WorkerName(...);

pub enum RemoteVersion {
    Current,
    Worker(WorkerName),
    Manifest(ManifestId),
}

pub struct RemoteRepository<B> { /* private */ }
```

Representative operations are:

```rust
impl<B> RemoteRepository<B> {
    pub fn status(&self) -> Result<RemoteStatus, RepositoryError>;
    pub fn resolve(&self, version: &RemoteVersion) -> Result<ManifestInfo, RepositoryError>;
    pub fn push_worker(
        &self,
        worker: &WorkerName,
        source: &Store,
        options: &RemoteWriteOptions,
    ) -> Result<PublishOutcome, RepositoryError>;
    pub fn aggregate_workers(
        &self,
        workers: &[WorkerName],
        options: &RemoteWriteOptions,
    ) -> Result<PublishOutcome, RepositoryError>;
    pub fn pull_to(
        &self,
        version: &RemoteVersion,
        destination: &std::path::Path,
        options: &PullOptions,
    ) -> Result<Store, RepositoryError>;
}
```

Only complete `ManifestId` values are accepted. `ManifestInfo` reports logical identity, physical config, segment count, physical bytes, and overall key bounds without exposing raw wire structs.

The repository implementation may access private `Store` internals because it lives in the same crate. It must not add public segment-path enumeration or raw manifest accessors merely for synchronization.

## Publication Order

Every publication follows one order:

1. Resolve and validate the target ref and repository root.
2. Capture source snapshots.
3. Produce or reuse verified main-only segment artifacts.
4. Stream every absent segment object to immutable storage.
5. Encode and upload the immutable manifest object.
6. Encode the next ref revision.
7. Compare-and-swap the target ref.

A failure before step 7 leaves the old ref visible. A compare-and-swap conflict also leaves the old candidate invisible. Uploaded immutable objects may be reused by a recomputed candidate.

Repository code must not publish a ref until it has confirmed that every referenced segment and the manifest object exist under their expected IDs.

## Worker Push

`push_worker` captures one stable local snapshot. If patches exist or the local codecs differ from repository config, it builds staged main-only output through the strict snapshot builder. Otherwise it may reuse complete existing main segments after verifying their bytes while streaming them across the trust boundary.

The worker ref is read once before candidate construction and updated with its exact provider token. One worker name has one logical writer; compare-and-swap still protects retries and accidental concurrent ownership.

## Aggregation

`aggregate_workers` resolves all requested worker refs and the current ref, then opens strict cursors over their manifests. It performs one k-way merge and one publication. It does not invoke pairwise `merge_from` repeatedly.

If the current ref changes before publication, aggregation resolves the new current head and rebuilds the logical union. It may reuse already uploaded segment objects, but it must rerun conflict detection against the new current snapshot.

The v1 manifest records only output content. Worker IDs and captured source manifest IDs are returned in `PublishOutcome` and may be logged by the caller; they are not persisted as an unbounded parent list.

## Pull

`pull_to` materializes the selected exact manifest into the local content cache. Creating a new local root may link or copy compatible verified segment bytes and assign fresh local segment IDs. Pulling into an existing root, or into different physical options, uses the strict snapshot builder and one atomic local publication.

Remote ref and manifest bytes never become local `MANIFEST` bytes. Local IDs, allocation state, and any temporary patch state remain local concerns.

## CLI

`scs` uses only public repository APIs. The initial commands are:

```text
scs remote status <remote>
scs remote push-worker <local-root> <remote> --worker <name>
scs remote aggregate <remote> --worker <name>...
scs remote pull <remote> <local-root> [--current | --worker <name> | --manifest <full-id>]
```

Mutating commands print the captured base ref revision, candidate manifest ID, and publication outcome. Compare-and-swap conflicts are distinct errors. No command parses remote catalog bytes or constructs provider object paths.

## Implementation Phases

### Phase 1: Domain And Provider Boundary

- Add identity, config, manifest, ref, and object-name validated types.
- Add canonical codecs and content-ID derivation.
- Add streaming immutable-object and bounded mutable-ref capabilities.
- Add an in-memory provider for deterministic protocol tests.
- Keep all existing local behavior unchanged.

### Phase 2: Materialization And Worker Push

- Add streamed temporary download and verified local cache publication.
- Add worker push with main-only normalization and codec conversion.
- Add pull into a new local root.

### Phase 3: Aggregation

- Reuse the strict snapshot builder for k-way aggregation.
- Add pull or merge into an existing local root.
- Test compare-and-swap rebasing and conflict atomicity.

### Phase 4: Public API And CLI

- Stabilize only the minimal synchronization API.
- Add the four initial `scs remote` commands.
- Verify that `scs` imports no private format, catalog, segment, or path modules.

## Validation

Domain and codec tests cover deterministic canonical encoding, exact round trips, malformed lengths, unknown IDs, duplicate references, invalid ranges, noncanonical order, trailing bytes, CRC failures, content-ID changes, and allocation limits.

Publication tests cover idempotent immutable insertion, missing-object rejection, upload-before-ref ordering, compare-and-swap conflicts, retry reuse, ref revision monotonicity, and failure atomicity.

Synchronization tests cover main-only worker publication, local codec conversion, complete segment verification, interrupted downloads, corrupted cache replacement, equal-record collapse, conflicting-value rejection, k-way order independence, and local v1 reopen after pull.

The existing default, all-feature, no-default, checksum, and compression test and clippy matrix remains required. Remote tests must not add provider SDKs to default local builds.

## Deferred Work

- History, parent DAGs, provenance objects, tags, pins, rollback, and diff.
- Listing, retention policy, garbage-collection planning, and deletion.
- Abbreviated IDs.
- Provider SDK crates and credential configuration.
- Async APIs and cancellation semantics.
- Authenticated range reads and mmap-backed content-cache reads.
