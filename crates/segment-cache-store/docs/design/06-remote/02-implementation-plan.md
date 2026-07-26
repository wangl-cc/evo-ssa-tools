# Remote Repository Implementation Plan

## Scope

This plan describes future, unimplemented provider-neutral remote support. It adds an optional capture/checkpoint API, restore/pull, and ordered aggregation without changing existing local operation semantics or local v1 bytes. The [remote repository design](00-design.md) is the behavioral contract and the [remote version format](01-version-format.md) defines persistent bytes.

The direct HPC baseline remains one durable network-filesystem local store per worker, with no shared-writer root. Remote support is needed only when a checkpoint, transfer, or aggregate exchange is desired. Node-local scratch is optional performance storage and depends on checkpoint recovery, not on a provider-visible local path.

## Crate Boundary

The provider-neutral engine belongs in `segment-cache-store` behind an optional `remote-repository` feature so it can capture and pin local snapshots, validate segment structure, and materialize local components without exposing raw local `MANIFEST` bytes or paths. Provider SDKs, credentials, retries, async runtimes, and transport configuration belong in adapter crates.

The initial internal boundary is:

```text
remote/
  identity.rs     logical store identity
  manifest.rs     snapshot kind, component layout, and canonical codecs
  capture.rs      local visible snapshot capture and segment-handle pins
  object.rs       typed IDs and derived object names
  provider.rs     immutable streams and bounded ref CAS
  publish.rs      resumable uploads, verification, and publication order
  restore.rs      verified transfer and local materialization
  aggregate.rs    ordered source capture and aggregate construction
```

Do not generalize the local catalog into a backend trait. Extract only the segment-set builder that accepts an already validated, sorted, unique record stream and emits bounded staged segments. Local normalization, remote re-encoding, and aggregate construction own different input error and winner policies upstream of that shared physical writer.

## Domain Types And API

Validated private domain types include `SnapshotKind`, `RemoteManifest`, `AggregateSnapshot`, `WorkerCheckpoint`, `RemoteSegmentRef`, `WorkerPatchRef`, and `CurrentThenWorkers`. The kinds have separate validated payloads rather than optional tier fields on one ambiguous segment set.

The public API exposes operations, not paths or raw catalog bytes:

```rust
pub struct WorkerName(/* private */);
pub struct ManifestId(/* private */);
pub struct CapturedSnapshot(/* private */);
pub struct WorkerOrder(/* private */);

pub enum RemoteVersion {
    Current,
    Worker(WorkerName),
    Manifest(ManifestId),
}

impl Store {
    pub fn capture_checkpoint(&self) -> Result<CapturedSnapshot, Error>;
}

impl<B> RemoteRepository<B> {
    pub fn checkpoint_worker(
        &self,
        worker: &WorkerName,
        snapshot: &CapturedSnapshot,
        options: &CheckpointOptions,
    ) -> Result<PublishOutcome, RepositoryError>;
    pub fn aggregate(
        &self,
        workers: &WorkerOrder,
        options: &AggregateOptions,
    ) -> Result<PublishOutcome, RepositoryError>;
    pub fn restore_to(
        &self,
        version: &RemoteVersion,
        destination: &std::path::Path,
        options: &RestoreOptions,
    ) -> Result<Store, RepositoryError>;
}
```

`CapturedSnapshot` is created only by `Store::capture_checkpoint` and owns its immutable runtime snapshot plus segment pins until dropped. The checkpoint method accepts only that type and can publish only a `WorkerCheckpoint` to the named worker ref. `WorkerOrder` has validated constructors for caller order and canonical lexicographic order of a supplied worker set; it rejects duplicate names. `aggregate` captures `refs/current` itself, prepends it when present, then resolves workers in the validated order into the private `CurrentThenWorkers` plan. It produces only an `AggregateSnapshot` and may update only `refs/current`. These types make invalid kind/ref and current-precedence states unnatural. The CLI uses them without enumerating segment paths or parsing local manifests.

Provider adapters need only streaming immutable objects and bounded mutable refs. Immutable insertion is idempotent, ref conflict is distinct from transport failure, and listing or deletion is not a v1 requirement.

```rust
use std::io::{Read, Write};

pub trait ImmutableObjects {
    type Error: std::error::Error + Send + Sync + 'static;

    fn get_to(&self, id: &ObjectId, sink: &mut dyn Write) -> Result<GetResult, Self::Error>;
    fn put_if_absent_from(&self, id: &ObjectId, len: u64, source: &mut dyn Read) -> Result<PutResult, Self::Error>;
}

pub trait MutableRefs {
    type Error: std::error::Error + Send + Sync + 'static;

    fn get(&self, name: &RefName) -> Result<Option<RefRead>, Self::Error>;
    fn compare_and_set(&self, name: &RefName, expected: Option<&RefToken>, next: &[u8]) -> Result<RefUpdate, Self::Error>;
}
```

Repository orchestration creates and owns a bounded temporary sink, then passes it to `get_to`; the provider only streams bytes into that caller-owned sink. Retries create a fresh sink and open a new source rather than assuming a partially consumed stream can rewind. Do not combine local locks, rename, directory synchronization, object streaming, listing, and CAS in one generic backend trait.

## Verified Artifact Boundary

Provider bytes remain untrusted until repository orchestration has:

1. Streamed them into a bounded temporary file.
2. Compared actual and declared lengths.
3. Recomputed the complete `SegmentContentId`.
4. Validated header, footer, logical geometry, block structure, self-described compression frames, and exact range.
5. Synchronized and atomically published the verified file into the local content cache.

An internal consumable `VerifiedSegmentArtifact` carries the temporary path and validated reference between verification and publication. It is not a durable trust marker or public capability. A failed verification removes the temporary or corrupt cache entry. Ordinary local open remains lazy only after this strict boundary has published the file.

## Capture, Planning, And Restore

Capture obtains one visible local layout while preventing commit and garbage collection from invalidating its segment handles. A worker checkpoint retains valid Main and PatchSet components and uploads their exact verified bytes; an unsupported format or codec is a hard error rather than an implicit migration. Aggregate planning reuses an exact verified segment only when source precedence and range nonoverlap make the complete segment valid output, and re-encodes only logically affected ranges. Neither operation normalizes or mutates the source local store merely to export.

All synchronization inputs are strict. Transfer verifies bounded length, complete `SegmentContentId`, segment structure, physical codec, and exact key range before an object is trusted. Corrupt root metadata, manifests, refs, or segments are errors. A corrupt current aggregate cannot be bypassed with a worker source.

Restoring a `WorkerCheckpoint` validates Main nonoverlap, Patch ownership and containment, PatchSet ordering, and fixed file and byte budgets before materializing Main plus Patch components. Restoring an `AggregateSnapshot` validates and materializes Main-only components. Both paths allocate local IDs locally.

Creating a new local root may link or copy compatible verified segment artifacts before publishing a newly encoded local MANIFEST. `restore_to` rejects an already completed destination root; combining remote data with an existing cache is an explicit local merge or remote aggregate operation, not an implicit restore mode. Remote refs and manifests never become local catalog bytes.

The shared segment-set builder receives only a canonical sorted, unique record stream, writes bounded Main segments using target options, and returns staged verified artifacts. Local normalization supplies an ordinary corruption-as-absence stream. Remote re-encoding and aggregate construction first strictly verify every source, then apply explicit source precedence, and only then feed the builder. This keeps trust and winner policy at the boundary that owns them instead of hiding them inside a generic writer.

## Publication And Aggregation

Publication is resumable rather than globally atomic:

1. Validate the remote root and capture the local snapshot or source refs.
2. Verify all referenced inputs and create a reuse/copy/re-encode plan.
3. Upload and verify each immutable segment independently.
4. Upload the immutable manifest only after every referenced object exists.
5. Compare-and-swap the optional target ref last.

Only the ref is a latest-reader visibility boundary. Failed transfers leave unreachable immutable objects reusable by retries. If compare-and-swap fails, capture the relevant current ref again and replan rather than publishing a stale aggregate.

Aggregate construction validates every captured source before precedence. `CurrentThenWorkers` fixes current first, followed by caller order or lexicographic order of the explicitly supplied workers. It streams records left-biased: the first record for a key wins, later values are ignored, and output is unique. This operation is deterministic for its captured priority but noncommutative; it neither rejects differing values nor claims order independence. It produces a Main-only aggregate and uses `AggregateOptions` only for newly encoded segments.

When an immutable content-ID path contains incorrect bytes, v1 returns a hard `CorruptExistingObject` error and does not publish a ref. Automatic replacement and quarantine are outside the portable provider traits; an administrator or provider-specific repair tool may conditionally repair the path out of band, but only with bytes whose hash is the expected ID. Correct content remains immutable.

## Implementation Phases

### Phase 1: Format And Provider Boundary

- Add identity, explicit `SnapshotKind`, manifest, component, ref, and typed-object codecs.
- Enforce kind-specific containment, overlap, ordering, and budget validation.
- Add a deterministic in-memory provider with immutable streams and ref CAS.

### Phase 2: Local Capture And Checkpoint

- Add high-level capture with commit/GC-safe segment pins.
- Add worker checkpoint planning, resumable object upload, and optional worker-ref publication.
- Preserve Main-plus-Patch checkpoint layouts when valid.

### Phase 3: Restore And Aggregate

- Add strict verified restore for both snapshot kinds.
- Add ordered left-biased aggregation to bounded Main-only output.
- Replan aggregation on current-ref compare-and-swap failure.

### Phase 4: Public API And CLI

- Stabilize the narrow capture, checkpoint/push, restore/pull, and ordered aggregate API.
- Add `scs remote` commands solely through that public API.
- Verify that adapters and CLI do not depend on local paths or raw local catalog bytes.

The initial CLI shape is:

```text
scs remote status <remote>
scs remote checkpoint <local-root> <remote> --worker <name>
scs remote aggregate <remote> --worker <name>...
scs remote restore <remote> <local-root> [--current | --worker <name> | --manifest <full-id>]
```

Mutating commands report the captured base revision, candidate manifest ID, and publication outcome. A compare-and-swap conflict is a distinct error, not a transport retry.

## Validation

Format and domain tests cover canonical encoding, mixed supported segment codecs, unsupported codecs, unknown kinds, kind mismatch at a ref boundary, malformed limits, and checkpoint Main/Patch containment, overlap, and budget failures.

Capture and publication tests cover snapshot pins surviving concurrent commit or GC, interrupted checkpoint resume, independent immutable upload verification, manifest-after-objects ordering, hard failure on a corrupt existing object, and unreachable-object reuse.

Synchronization tests cover strict corrupt-current failure, checkpoint and aggregate restoration shape, destination-biased differing values, current-first and validated worker priority, unique aggregate output, mixed-codec segment reuse, exact checkpoint/restore bytes, aggregate re-encoding only for logically affected ranges, and no source-store normalization during export.

Concurrency tests cover compare-and-swap failure followed by current recapture and replan. The existing default, all-feature, no-default, compression, test, and clippy matrix remains required; remote support must not add provider SDKs to default local builds.

## Deferred Work

- Parent DAGs, history, provenance objects, tags, pins, rollback, and diff.
- Listing, retention policy, remote garbage collection, and deletion.
- Provider SDK crates, credential configuration, async APIs, cancellation, authenticated range reads, and mmap-backed cache reads.
