# Remote Repository Design

## Status

This document specifies a future provider-neutral remote repository for `segment-cache-store`. None of this behavior is implemented by the current local store.

Remote support adds a separate catalog and publication protocol. It does not change local `STORE`, `MANIFEST`, or segment v1 bytes. The companion documents define the [remote version format](remote-version-format.md) and the [implementation plan](remote-implementation.md).

## Purpose

The local store already publishes immutable segment files through a small manifest. A remote repository preserves that shape: segment and manifest objects are immutable, while small compare-and-swap refs select visible manifests.

The primary workflow is distributed computation caching:

1. Workers commit locally.
2. Each worker explicitly publishes one normalized main-only snapshot to its worker ref.
3. An aggregator strictly merges selected worker heads and the current remote head in one k-way pass.
4. The aggregator publishes one main-only manifest and advances `refs/current` with compare-and-swap.
5. Readers materialize a selected manifest into a local content cache and use the existing local read engine.

Remote state is a phase-boundary exchange format, not the hot execution backend and not a continuously synchronized LSM level.

## V1 Decisions

- Keep local execution and hot reads in `Store`.
- Make push, pull, and aggregation explicit operations.
- Keep local and remote catalog wire formats independent.
- Reuse the existing BLAKE3 `SegmentContentId` for complete segment bytes.
- Address immutable remote manifests by a BLAKE3 content ID.
- Keep logical store identity separate from physical checksum and compression policy.
- Require every remote manifest to contain a main-only, sorted, non-overlapping segment set.
- Collapse equal keys only when value bytes are identical; differing values abort publication.
- Upload immutable objects before publishing the ref that makes them visible.
- Use only complete IDs in the v1 API and CLI.
- Stream segment upload and download through files or readers and writers; do not require whole segments in memory.
- Let `scs` call only public repository APIs.

## Non-Goals

- A general-purpose remote database.
- Low-latency point reads directly from cold object storage.
- POSIX filesystem emulation over an object provider.
- Automatic publication of every local commit.
- A generic storage trait shared by local rename and remote compare-and-swap.
- History traversal, parent DAGs, tags, pin leases, rollback, logical diff, abbreviated IDs, or destructive remote garbage collection in v1.
- Authenticated range reads before a range-authentication format exists.

## Architecture

```text
application or scs
        |
public remote repository API
        |
repository protocol: validate, push, pull, aggregate, publish
        |
immutable object stream + mutable ref compare-and-swap
        |
provider adapter
```

The provider-neutral repository engine may live in `segment-cache-store` behind an optional `remote-repository` feature so it can reuse private segment readers, writers, and the strict snapshot builder. Provider SDKs and their dependency trees belong in adapter crates.

The local catalog is not generalized into a backend-independent catalog. Local publication owns local IDs, patches, locks, rename, and directory synchronization. Remote publication owns content IDs, immutable objects, and conditional ref updates.

## Identity And Physical Policy

`StoreIdentity` answers one question: whether two snapshots represent the same logical cache namespace. It covers key geometry, value layout, and caller metadata. It does not include block checksum or value compression choices.

Each remote root also has one v1 `RepositoryConfig` that selects the physical checksum and compression used for remote segment objects. The config is immutable for v1, but it is not part of logical identity. A future config migration may rewrite physical objects without creating a new logical cache namespace.

Publishing a local store whose codec differs from the repository config requires rebuilding its visible records with the repository config. Reusing complete segment objects is valid only when the bytes already satisfy the target config and their ranges remain non-overlapping in the output.

## Object Model

The portable object layout is:

```text
<root>/
  store/identity
  store/config
  refs/current
  refs/workers/<worker-name>
  manifests/<manifest-id>.manifest
  segments/<segment-content-id>.seg
```

Identity and config are bounded root metadata. Segment and manifest objects are immutable. Refs are bounded mutable objects updated with provider compare-and-swap tokens.

Object listing is not part of visibility, resolution, push, pull, or aggregation. A future maintenance protocol may add listing and deletion without changing ordinary publication semantics.

## Worker Publication

A worker publication:

1. Captures one stable local visible snapshot.
2. Validates logical identity against the remote root.
3. Normalizes local patches into staged main-only output without mutating the live local root.
4. Re-encodes output when the local physical policy differs from the repository config.
5. Streams missing segment objects to immutable storage while checking length and content ID.
6. Uploads the immutable manifest object.
7. Advances only that worker's ref with compare-and-swap.

Each worker owns its ref name. Workers do not contend on `refs/current` during normal execution.

## Aggregation

Aggregation captures `refs/current` and all selected worker refs, then performs one strict k-way merge. It emits a bounded sequence of main segments using the repository config and publishes one immutable manifest.

The operation follows these rules:

- Identical records collapse to one output record.
- Equal keys with different values return a conflict and leave `refs/current` unchanged.
- Missing, malformed, or hash-mismatched input objects are hard aggregation errors.
- Existing segment objects are reused only when their complete ranges remain non-overlapping and their physical config matches.
- Otherwise records are streamed through the same strict snapshot builder used by local cross-store merge.
- A compare-and-swap conflict causes the aggregator to capture the new current head and recompute logical intent; it never overwrites the new head with a stale candidate.

V1 manifests are content snapshots and contain no parent list. Exact source provenance belongs in operation logs or a future history object rather than making arbitrary worker fan-in part of the snapshot format.

## Pull And Materialization

A pull:

1. Resolves `refs/current`, one worker ref, or one complete manifest ID.
2. Fetches and verifies the immutable manifest.
3. Streams each missing segment into a temporary local file.
4. Verifies length, complete `SegmentContentId`, header, footer, and range invariants before atomic cache publication.
5. Builds or replaces a local root using local v1 publication rules.

The remote manifest is never copied into local `MANIFEST`. Local segment IDs and publication state are assigned locally. If the destination's physical options differ from the repository config, pull rebuilds records through the strict snapshot builder instead of linking incompatible segment bytes.

## Concurrency And Visibility

Immutable insertion is idempotent. A manifest becomes discoverable only after all referenced segment objects exist. A ref update is conditional on the exact provider token returned by the preceding ref read.

If compare-and-swap fails, the candidate manifest is not visible through that ref. Its immutable objects remain safe to reuse in a retry and may become collectible when a future maintenance protocol exists.

## Failure Semantics

Failure handling depends on the operation boundary:

| Boundary | Missing or corrupt segment behavior |
| --- | --- |
| Push, pull, aggregation, or publication | Hard error; do not publish a new ref |
| Verification while filling the local content cache | Remove the temporary or corrupt cache entry and report failure or retry |
| Ordinary reads from an already opened local cache store | Preserve the existing corruption-as-miss contract; never return unverified bytes |

Malformed identity, config, manifest, or ref bytes are always hard repository errors because they define namespace, interpretation, or visibility.

Content IDs detect accidental corruption and object substitution. They do not authenticate who is allowed to update a ref. Authorization remains the responsibility of authenticated transport and provider access control.

## Deferred Work

- History and source-provenance objects.
- Tags and retention leases.
- Rollback and physical or logical diff APIs.
- Abbreviated ID resolution.
- Listing-based remote garbage collection and deletion.
- Async repository APIs.
- Authenticated range reads and mmap-backed local materialization.
