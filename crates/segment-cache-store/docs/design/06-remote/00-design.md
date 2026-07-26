# Remote Repository Design

## Status

This is a future, unimplemented design for provider-neutral remote exchange in `segment-cache-store`. It deliberately revises the pre-release remote-v1 target. The current crate implements only the local store.

Remote support is an explicit checkpoint, transfer, restore, and aggregation facility. It does not alter local `STORE` or `MANIFEST` v1 bytes, make a remote provider the hot execution backend, or continuously synchronize LSM state. The companion documents define the [remote version format](01-version-format.md) and [implementation plan](02-implementation-plan.md).

## HPC Baseline

The direct HPC baseline is one durable network-filesystem `Store` root per worker. Multiple workers never write the same root. This is correct without remote support and remains the preferred simple deployment when the filesystem supplies the local store's required positioned I/O, atomic rename, directory durability, and advisory-lock behavior.

Node-local scratch is optional performance storage. A worker that uses scratch must checkpoint or publish through `segment-cache-store` before scratch loss matters. Neither the CLI nor a provider adapter receives local paths, raw `MANIFEST` bytes, or local allocation details.

Checkpoint capture is segment-aware and performed by a high-level public API inside this crate. It captures one local visible snapshot and pins its referenced segment handles against concurrent commit and garbage collection until transfer finishes. A capture cannot observe a mixture of local publication states.

## Snapshot Products

Remote manifests have an explicit validated `SnapshotKind`; kind is never inferred from a ref path.

- `WorkerCheckpoint` is a restart product. It may preserve a valid Main component and its owned `PatchSet` components, avoiding whole-segment rewrites before a worker restarts.
- `AggregateSnapshot` is a distribution product. It contains only Main components, sorted and non-overlapping, and is the only kind permitted at `refs/current`.

`WorkerCheckpoint` references use separate Main and Patch arrays. Each Patch names exactly one Main owner, so tier is represented by the containing field rather than a per-reference discriminant. Main ranges are globally nonoverlapping; each Main-owned PatchSet is sorted, nonoverlapping, contained by its owner, and subject to the same fixed file and byte caps as the local target. `AggregateSnapshot` references remain Main-only and globally sorted and nonoverlapping.

Restore validates identity and all kind-specific containment, ordering, range, and budget invariants before materializing local state. Restoring a checkpoint may materialize Main plus Patch components; restoring an aggregate materializes Main-only. Local segment IDs are always allocated locally.

## Identity And Objects

`StoreIdentity` identifies a logical cache namespace using key and value geometry plus the same opaque caller metadata used by the local store. That metadata is where the caller encodes schema and semantic namespace versions. Identity does not include platform or build fingerprints by default. A semantic change requires a new identity; acceptable platform-level numerical variation may merge because scientific numerical equivalence is decided above storage.

Remote roots contain one immutable logical identity, immutable segment and manifest objects addressed by complete content IDs, and small mutable refs updated by compare-and-swap:

```text
<root>/
  store/identity
  refs/current
  refs/workers/<worker-name>
  manifests/<manifest-id>.manifest
  segments/<segment-content-id>.seg
```

Content IDs name complete immutable bytes. A byte sequence found at a content-ID path that does not hash to that ID is corrupt, not an immutable object. Portable v1 operations return a hard error and publish no ref. A provider-specific administrator may conditionally replace or quarantine the bad path out of band; immutability means correct content is not changed, not that corrupt bytes become permanent.

V1 has no parent DAG, history traversal, tags, pin leases, listing, or remote garbage collection. Refs are visibility pointers, not history records.

## Architecture

```text
application or scs
        |
public capture, checkpoint, restore, and aggregate API
        |
repository protocol: validate, plan, transfer, publish
        |
immutable object streams + mutable ref compare-and-swap
        |
provider adapter
```

The provider-neutral engine may live behind an optional `remote-repository` feature so it can reuse private segment readers, writers, capture pins, and strict builders. Provider SDKs belong in adapter crates. The local catalog is not generalized into a backend-independent catalog: local publication owns local IDs, patches, locks, rename, and directory synchronization; remote publication owns content IDs, immutable objects, and conditional refs.

## Planner And Publication

The remote planner is driven by the captured local layout. Worker checkpoint preserves every verified segment byte-for-byte and fails on an unsupported segment format or codec. Compression is self-described per segment, so supported codecs may coexist in one remote manifest. Aggregate construction reuses exact verified segments only where precedence and nonoverlap make them complete output ranges; it re-encodes logically affected ranges with `AggregateOptions`. Export never mutates or normalizes the local `Store` merely to make a remote snapshot.

Publication is resumable, with this order:

1. Validate the repository root and capture the input snapshot or refs.
2. Verify every referenced input object and plan reuse, copy, or re-encoding.
3. Upload and verify immutable segment objects independently.
4. Upload the immutable manifest after all of its objects exist.
5. Optionally compare-and-swap the target ref last.

Only the ref update is the visibility boundary for latest readers. An interrupted transfer leaves unreachable immutable objects that a later operation may reuse. A failed ref compare-and-swap recaptures the relevant current ref and replans; it never publishes stale logical intent.

## Trust Boundary

Push, checkpoint, import, pull, and aggregate verify every referenced input before use: declared length, `SegmentContentId`, segment structure, range, and self-described codec. Identity, ref, and manifest parsing is likewise strict. A corrupt `refs/current` input is a hard error and is never replaced by a lower-priority worker result.

This strict boundary is distinct from ordinary reads of an already opened local cache store, which retain the local corruption-as-miss behavior. Synchronization must not launder unverified bytes into a new snapshot.

Failed download verification removes the temporary or corrupt local content-cache entry before reporting failure or retrying. A verified immutable object may then be published into that cache atomically; a partially verified file is never exposed as trusted content.

Content IDs detect corruption and substitution but do not authenticate who may publish a ref. Authenticated transport, provider authorization, and any external mechanism that establishes a trusted remote manifest remain deployment responsibilities rather than storage-format claims.

## Pull And Materialization

Pull resolves a current ref, worker ref, or complete manifest ID; verifies the manifest and every segment into bounded temporary files; then links or copies those exact segment bytes into a new local store and publishes a newly encoded local MANIFEST with fresh local IDs. An unsupported segment format or codec is a hard error. Remote manifest bytes never become local `MANIFEST` bytes. Restoring into an already completed local root is rejected rather than silently replacing or merging its cache; callers use explicit local merge or remote aggregation for union semantics.

## Aggregation

Aggregation is deterministic but deliberately noncommutative. It captures an ordered list of source `ManifestId`s after strictly validating all selected inputs. `refs/current`, when present, is first; the explicitly supplied duplicate-free worker set follows either in caller order or in lexicographic encoded `WorkerName` order. For each key, the record from the first source in that precedence wins and output contains the key once.

Source corruption is an error before precedence is applied. Aggregation does not compare differing values for conflicts and does not claim k-way order independence. It produces an `AggregateSnapshot`, using `AggregateOptions` for newly encoded output, then may conditionally advance `refs/current`.

## Public Boundary

The crate exposes high-level capture, checkpoint/push, restore/pull, and ordered aggregate operations. Provider adapters expose only immutable streaming objects and bounded ref compare-and-swap. `scs` uses these public operations and does not inspect local paths or raw local/remote catalog bytes.

## Non-Goals

- A general-purpose remote database or POSIX filesystem emulation.
- Low-latency remote point reads.
- Automatic publication after every local commit.
- A generic trait unifying local rename, locks, and remote provider operations.
- Parent DAGs, history, tags, rollback, listing, destructive remote GC, abbreviated IDs, or authenticated range reads in v1.
