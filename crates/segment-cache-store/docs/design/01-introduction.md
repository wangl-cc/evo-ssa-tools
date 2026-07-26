# Introduction

`segment-cache-store` is a narrow persistent backend for recomputable computation caches with fixed-width ordered keys and opaque serialized values. It is batch-first and insert-only; it is not a general-purpose database.

## Purpose And Scope

The normal workflow is to look up a sorted batch, compute its misses, and publish the new records for later reuse. The store protects the stored byte mapping, not the scientific correctness of those bytes. Scientific equivalence, numerical tolerance, and platform-dependent numerical variation belong above this layer. A semantic model change requires caller metadata or a key-namespace version; a checksum-valid but scientifically wrong entry requires rebuilding that namespace. There is no per-key invalidation or repair API.

The design is optimized for fixed-width keys whose lexicographic byte order is the desired query order, ordered batch lookup, complete ordered iteration for export and analysis, immutable segment files as synchronization units, and batch publication rather than individual durable writes.

The target deliberately excludes transactions, deletes, tombstones, update-in-place, concurrent writers, a WAL, background compaction, general query features, and arbitrary tiny durable commits. One cooperating logical writer publishes batches; sparse single-record gap writes that create tiny Main files are explicitly non-target until a separate convergence policy is defined.

Values are recomputable, so classified data corruption is logically absent in ordinary cache paths. The store must never return corrupted bytes as a valid hit.

The crate and format remain experimental. Until a format is released with a compatibility promise, rebuilding the cache from recomputable source data is the migration strategy.

## Logical Model

A visible logical component is exactly one `Main` segment plus its ordered `PatchSet`.

- Main segments are globally sorted, nonoverlapping, and hard bounded by the Main format cap.
- Each Patch belongs to exactly one Main and has a key range contained in that Main's range.
- Patches in one PatchSet are sorted and nonoverlapping. A key has point depth at most one: one healthy Main candidate and at most one healthy Patch candidate.
- Multiple disjoint patches for a Main are allowed. Patch count still affects open, manifest, and scan costs even though point depth is one.
- A visible key has one value. Normal publication never creates duplicates.

There is no interval-connected or transitive normalization-component model in the target.

## Batch And Duplicate Contracts

`WriteBatch` input is sorted and contains no duplicate keys. Duplicate keys inside one batch are invalid input. During normal publication, a healthy, already-visible destination key wins; an incoming record with that key is skipped without comparing value bytes. Therefore a normal commit never reports `KeyConflict` for a differing incoming value.

An abnormal duplicate in persisted data is handled deterministically: a healthy Main value wins ordinary reads and normalization, and explicit verification reports `DuplicateVisibleKey`. If the Main is absent or corrupt, a healthy Patch may supply the key. This preserves cache availability without turning tier order into a caller-facing conflict policy.

Cross-store and remote aggregation use the same storage-level principle with explicit source precedence: the first healthy source wins and later values are not compared. This tolerates scientifically acceptable cross-platform or cross-version numerical differences without making storage responsible for numerical equivalence. Callers that need distinct stochastic results must encode the seed, replicate, or other logical identity in the key.

## Caller Contract

- **Stable key encoding.** The same logical input must produce the same fixed-width key bytes wherever the namespace is used. The store compares bytes lexicographically and does not understand the key's internal fields.
- **Semantic namespace versioning.** Caller metadata or the encoded key namespace must change when a scientific or serialization change makes old results semantically invalid. Storage checksums cannot detect a checksum-valid but scientifically wrong result.
- **Validated batches.** A batch must contain keys of the store's fixed width, values compatible with its value layout, and no duplicate keys. The store may sort an unsorted batch before preparing it, but the internal publication boundary receives a canonical sorted, duplicate-free representation.
- **Single logical writer.** One cooperating writer mutates a root at a time. The advisory lock rejects a second cooperating writable open; read-only handles may coexist.
- **Managed immutable files.** Published segment files must not be externally modified while handles or captured snapshots use them. External mutation without the writer protocol is outside the local safety contract.

## Terminology

- `Key`: a fixed-width byte string ordered lexicographically.
- `Value`: opaque serialized bytes associated with a key.
- `Entry`: a caller-provided key/value pair before persistence.
- `Record`: a key/value pair returned from storage.
- `Main`: an immutable, sorted segment in the globally nonoverlapping Main set.
- `Patch`: an immutable, sorted segment owned by one Main and contained in that Main's range.
- `PatchSet`: the ordered, nonoverlapping patches owned by one Main.
- `Normalization`: foreground replacement of one Main and its PatchSet with bounded Main segments.
- `STORE`: immutable namespace identity and caller metadata.
- `MANIFEST`: the atomic snapshot of visible Main-owned PatchSets and logical schema.
- `SegmentContentId`: BLAKE3 identity of a complete segment file for a trust-boundary crossing.

## Implementation Status

This design specifies the target v1 contract. The current pre-release implementation retains a flat main-then-patch manifest and permits one patch per interval-connected component. Its publication, read, and normalization behavior is called out in current implementation notes where later chapters identify a difference; the target remains format version `1` because no format has been released.
