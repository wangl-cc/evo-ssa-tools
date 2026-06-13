# Design

## Purpose

`segment-cache-store` is a storage backend for computation-cache workloads, not a general-purpose database.

The intended workload has these properties:

- keys are fixed-width byte strings
- key order is meaningful and query order follows key order
- values are already serialized and opaque to the store
- each store declares one value layout: variable-length values or fixed-length values
- writes publish immutable sorted segments in batches
- ordered batch lookup is the hot path
- corruption should degrade to cache misses, not incorrect results
- store roots are synchronized between devices by copying segment files

This backend intentionally avoids functionality that is expensive but not useful for this workload:

- no transactions
- no deletes
- no update-in-place
- no background compaction: a segment is rewritten only when a commit interleaves with its range
- no concurrent writers: one writer at a time, with no merging of concurrent writer state
- no WAL-based recovery

## Status

This crate is experimental. The on-disk format may change without migration support; when the format changes, existing stores are rebuilt from scratch. There is no v1-to-v2 compatibility requirement.

Sections and paragraphs marked **Decided** record accepted design. Everything unmarked describes the current implementation.

Implementation status (see [Implementation Sequencing](#implementation-sequencing) for the full plan):

- **Implemented**: replacing-manifest commits, bounded L0 patch segments with normalization, the winner rule, dead-entry dropping, the advisory writer lock, open-time garbage collection, and the `STORE`-last creation order.
- **Pending**: content-addressed segment identity, generations, and the cross-device sync protocol (the catalog revision that carries them). Paragraphs describing only these still carry a **Decided** marker.

## Core Invariants

Permanent invariants:

- one store root has one fixed key length, one value layout, and one caller-defined metadata namespace
- published segment files are immutable; a segment file's bytes never change after publication
- a root's `MANIFEST` is the sole source of visibility for that root
- missing or corrupted data is treated as absent, never as valid output
- main-tier segment ranges are sorted and globally non-overlapping
- patch-tier segment ranges are bounded and may overlap main-tier or other patch-tier ranges
- segments whose ranges receive no new writes are never rewritten
- all stored copies of one key are semantically interchangeable (see [Caller Contract](#caller-contract))

The following former restrictions are now lifted in the implementation:

- commits are no longer insert-only: a batch whose range intersects main-tier data can publish to the patch tier or trigger normalization (see [Steady-State Write Design](#steady-state-write-design))
- manifest entries whose segment files are missing no longer reserve their ranges forever: they are dropped at the next manifest publication

## Caller Contract

The store enforces key width and value layout. The contracts below cannot be checked by the store and are the caller's responsibility.

- **Stable canonical key encoding.** Key encoding is external; the store only compares keys lexicographically. Any cross-device use additionally requires that the same logical input encodes to the same key bytes on every platform (`ssa-workflow` provides this via its canonical input encoding). An unstable key encoding fragments the cache - spurious misses and duplicate work - but does not corrupt it.
- **Interchangeable values.** Every value ever committed for one key must be semantically interchangeable with every other. Byte equality is not required and is not assumed: floating-point library differences across platforms and codec or compression version differences legitimately produce different bytes for equivalent values. The store may return or retain any one copy.
- **Single logical writer.** At most one process writes to a root at a time (see [Process Model](#process-model)).

## Terminology

The storage design uses the following terms consistently. Code names should follow these meanings unless a local type documents a narrower role.

### Logical Data

- `Key`: the fixed-width byte string used for ordering and lookup. Key encoding is external to this crate; the store only compares keys lexicographically.
- `Value`: the opaque serialized byte string associated with one key. Value serialization is external to this crate.
- `Entry`: a logical key/value pair before it is encoded into a segment file. Write batches and segment writers consume entries.
- `Record`: a key/value pair as decoded from visible stored data. Readers, lookup sessions, and scan cursors produce records. A record can be treated as missing if its containing block is corrupt.
- `Copy`: one stored record for a key that may also be stored elsewhere. Copies of one key are semantically interchangeable by caller contract; their bytes may differ.

`Entry` and `Record` are intentionally different words: an entry is caller input to the encoder, while a record is storage output from decoded bytes. Unqualified `Entry` should mean logical input data; other entry-like structures must be qualified, such as `Manifest entry` or `Block index entry`.

### Tiers

- `Main tier (L1)`: the sorted, globally non-overlapping segment set.
- `Patch tier (L0)`: a small bounded segment set that may overlap the main tier and each other.
- `Normalization`: a replacing commit that merges patch-tier segments and the intersecting main-tier segments into new main-tier segments.

### Store Catalog

- `Store root`: the directory containing one cache namespace.
- `Metadata`: caller-defined compatibility bytes persisted in `STORE`. Metadata can describe namespace, schema, codec, or any other caller-level compatibility contract.
- `STORE`: the stable store descriptor. It records persistent identity such as metadata, key length, and value layout. It is created once and is not the atomic publish point for data.
- `MANIFEST`: the atomic visible-data snapshot. It lists the currently visible segments.
- `Manifest entry`: one visible segment range in `MANIFEST`, identified by an opaque segment reference plus `min_key` and `max_key`.
- `Generation`: a per-root monotone counter incremented by every manifest publication. Generations order snapshots of one root; they have no meaning across roots. **Decided**, not implemented.

### Physical Storage

- `Segment`: an immutable file containing a sorted key range. Segments are the file-level sync unit and the manifest visibility unit.
- `Segment header`: fixed-size metadata at the start of a segment. It identifies segment format and store geometry needed before decoding the rest of the file.
- `Segment footer`: variable-size metadata at the end of a segment. It is the completion marker and owns the sparse block index.
- `Block`: the complete physical read and checksum unit inside a segment. A block contains key bytes, value-layout bytes, padding, a block footer, and the block checksum.
- `Block index`: sparse segment-footer metadata with one entry per block. It maps each block's first key to its absolute file offset, physical length, and record count.
- `Block footer`: fixed-size block-local decoding metadata stored at the end of a block. It records the common key prefix length, value payload location, and block checksum.

### Block Internal Layout

- `Region`: a contiguous byte range inside a block owned by one layout. A region can contain payload bytes, auxiliary index data, or both.
- `KeyRegion`: the region owned by key layout. It currently stores the common key prefix followed by fixed-width key suffixes.
- `ValueRegion`: the region owned by value layout. Variable-value blocks store a `ValueIndex` followed by a `ValuePayload`; fixed-value blocks store only a `ValuePayload`.
- `Index`: auxiliary bytes used to locate payload. Index bytes are not returned to callers.
- `ValueIndex`: the variable-value offset table. It has `record_count + 1` offsets, where the final offset is the value payload length.
- `Payload`: stored user-data bytes after layout metadata has been removed. Do not use `payload` for generic format bodies such as segment footers or manifests.
- `ValuePayload`: the packed value bytes returned to callers. `payload_offset` and `payload_len` always refer to this payload, not to the whole `ValueRegion`.

These terms intentionally avoid names like `area` or `after_keys`. `Area` is too vague, and `after_keys` describes the current neighboring layout rather than the value layout's own responsibility.

## Directory Layout

Each store directory contains:

```text
root/
  STORE
  STORE.tmp
  MANIFEST
  MANIFEST.tmp
  LOCK
  segments/
    segment-00000000000000000000.seg
    segment-00000000000000000001.seg
    segment-00000000000000000002.seg.tmp
```

`STORE` is the stable namespace descriptor. It changes only when the store is created. `MANIFEST` is the atomic visible segment set. `LOCK` is the stable advisory-writer lock file and is never atomically replaced. Segment files that are not referenced by `MANIFEST` are ignored. Temporary files are sibling files created by adding a `.tmp` extension to their final target, and are implementation details of atomic publication; open/read paths do not scan directories to discover visible data.

**Creation order.** Store creation writes `MANIFEST` first and `STORE` last; `STORE` is the creation completion marker. `Store::create` succeeds whenever `STORE` is absent, overwriting any leftover `MANIFEST` from an aborted creation; `Store::open` requires both files. A creation that crashes between the two writes leaves a root with no `STORE`, which `create` safely re-creates.

`STORE` is line-oriented text, not JSON. Binary metadata is encoded as lowercase hex.

The `STORE` format is:

```text
segment-cache-store store v1
version=1
metadata=<hex>
key_len=<u32, decimal>
value_len=<u32, decimal>
```

`value_len=0` means variable-length values. Any non-zero value means fixed-length values of exactly that many bytes. `STORE` is authoritative for `key_len`; the copy in `MANIFEST` is a cross-check only.

The `MANIFEST` format is a binary snapshot:

```text
magic[4]              # b"SCSM"
version:u32           # 1
key_len:u32
next_segment_id:u32
segment_count:u32
manifest_entries[segment_count]
crc32c:u32            # covers all previous manifest bytes
```

Each manifest entry is:

```text
segment_id:u32
tier:u8                 # 0 = main, 1 = patch
min_key[key_len]
max_key[key_len]
```

`segment_id` derives the file name `segments/segment-<segment_id>.seg`; file names are opaque identities and do not define key order. Manifest entries are ordered as all main-tier entries sorted by `min_key`, followed by patch-tier entries sorted by `(min_key, segment_id)`. On open, the store rejects malformed `STORE` or `MANIFEST` files, duplicate segment ids, unsorted or overlapping main-tier ranges, patch entries that precede main entries, and `next_segment_id` values that could reuse an existing segment id.

Missing manifest-referenced segment files are tolerated as miss space; their entries are dead entries, dropped at the next manifest publication (see [Steady-State Write Design](#steady-state-write-design)).

The decided revision of this format is described in [Catalog Revision](#catalog-revision).

## Segment File Layout

Each segment file contains:

```text
SegmentHeaderV1
Data Blocks
SegmentFooterV1, including block index
```

All integer fields are little-endian.

The fixed 24-byte segment header is:

```text
magic[4]             // b"SCSG"
format_version: u32  // 1
key_len: u32
value_len: u32       // 0 = variable values, >0 = fixed value length
reserved: u32        // must be 0
header_crc32c: u32   // crc32c over the previous 20 bytes
```

The segment footer body is variable-length because it includes the fixed-width min and max keys plus the sparse block index:

```text
record_count: u64
min_key[key_len]
max_key[key_len]
block_count: u32
block_index_entries...
footer_len: u32        // footer body length
footer_crc32c: u32    // crc32c over footer body | footer_len
```

Each block index entry stores:

```text
first_key[key_len]
block_offset: u64
block_len: u32
record_count: u32
```

Block offsets are absolute file offsets. The first data block starts at byte offset `24`.

The reader opens a segment by validating the header, reading the footer from the end of the file, validating the footer and block index, then loading the sparse block index into memory. If the header, footer, or block index is invalid, the whole segment is ignored.

Segment encoding is deterministic: the same sorted entries written with the same writer parameters produce byte-identical files. This property is load-bearing for content-addressed identity and sync convergence; writer changes must preserve it or be treated as format changes.

## Block Layout

Blocks do not have a header. Block length and record count come from the segment footer's block index. Block-local decoding metadata is stored in a fixed-size footer at the end of each block, with CRC32C as the final field.

The block footer records the `ValuePayload` position because that is the part needed by both fixed and variable value layouts:

```text
block body
padding
prefix_len: u32
payload_offset: u32
payload_len: u32
block_crc32c: u32     // crc32c over all previous bytes in this block
```

`prefix_len` is the block-local common key prefix length. `payload_offset` and `payload_len` describe the packed value payload inside the block. Padding appears before the footer so the footer remains at `block_len - 16`.

`OpenOptions::with_block_checksum_verification(false)` disables data-block checksum verification for read-only benchmarking against engines that do not validate user value bytes on read. Writable opens reject this option so corrupted bytes cannot be merged into freshly checksummed replacement segments. This is not the default cache-safe mode: with verification disabled, corrupted value bytes may be returned as hits.

Each block strips the common prefix shared by its first and last key. Because keys are sorted, this prefix is shared by every key in the block. The suffix table remains fixed-width, so the reader can still binary-search by record index.

Variable-value blocks store the common prefix once, then a contiguous fixed-width key suffix table, then value offsets, then packed value bytes. The offset table has `record_count + 1` entries; the last entry is the value payload length, so value `i` is `value_offsets[i]..value_offsets[i + 1]` without a last-record branch.

```text
KeyRegion:
  key_prefix[prefix_len]
  key_suffixes[record_count * (key_len - prefix_len)]
ValueRegion:
  ValueIndex:
    value_offsets[record_count + 1]: u32
  ValuePayload:
    values...
padding...
prefix_len: u32
payload_offset: u32
payload_len: u32
block_crc32c: u32
```

Fixed-value blocks omit the value index:

```text
KeyRegion:
  key_prefix[prefix_len]
  key_suffixes[record_count * (key_len - prefix_len)]
ValueRegion:
  ValuePayload:
    values[record_count * value_len]
padding...
prefix_len: u32
payload_offset: u32
payload_len: u32
block_crc32c: u32
```

Lookup binary-searches the reconstructed fixed-width key table. Variable layout then uses the matching value offset pair. Fixed layout computes `value_start = payload_offset + index * value_len`.

Block sizing rules:

- blocks smaller than the target block size are padded up to the target size
- oversized blocks are written at their natural size and are not padded further

The target block size is a writer policy, not a read compatibility invariant. Existing segments store their physical block lengths in the block index, so a store can be reopened with a different target block size and future segments can use the new size.

## Format Limits

The format uses `u32` for most lengths and counts. The resulting envelope, enforced as write-time errors:

- `key_len`: 1 to `u32::MAX` bytes
- one block: at most 4 GiB physical length; value offsets inside a block are `u32`, so one block's value payload is also capped at 4 GiB
- one segment: at most `u32::MAX` blocks in the block index and `u32::MAX` records per block; total records per segment are counted as `u64`
- one manifest: at most `u32::MAX` segment entries

These limits are far above the intended workload shape (16 KiB-class blocks, MiB-class segments) and exist to keep the format compact, not to be approached.

## Read Path

`fetch_one` binary-searches the main tier by `min_key`, checks that the candidate segment also satisfies `key <= max_key`, then probes any patch segments whose ranges contain the key. If multiple copies exist, the lexicographically smallest value bytes win.

Ordered lookup is the main hot path. The low-level APIs are:

- `Store::contains_many_ordered`
- `Store::fetch_many_ordered`
- `Store::visit_many_ordered`
- `Store::lookup_session`
- `OrderedLookup::contains_many`
- `OrderedLookup::fetch_many`
- `OrderedLookup::visit_many`

`visit_many` is the lowest-level ordered-read API. It lets callers consume borrowed value slices instead of forcing a `Vec<u8>` allocation for every hit.

Ordered lookup sweeps sorted keys against the main tier and, when patches exist, sweeps each patch segment with block reuse before applying the same winner rule. Within a segment it keeps the current block loaded and consumes all keys that fall inside that block range before advancing. This reduces repeated block loads and makes large ordered hit streams much cheaper than independent lookups.

`iter_all()` returns all visible records in order. `range(start, end)` returns the half-open range `[start, end)`. If no patch segments are visible, scan cursors concatenate main-tier segment cursors in manifest order. If patches exist, scan uses a k-way merge across main and patch cursors and deduplicates copies with the winner rule. They stream at block granularity and do not materialize the full scan result before returning.

## Write Path

The public write path is:

- `Store::begin_batch`
- `WriteBatch::push`
- optional `WriteBatch::mark_sorted`
- `Store::commit_batch`

`commit_batch`:

1. validates key lengths
2. validates fixed value lengths when fixed layout is enabled
3. sorts the batch if needed
4. rejects duplicate keys inside the batch
5. finds the contiguous run of main-tier segments whose ranges the batch intersects
6. if no main-tier range is touched, writes the batch directly as main-tier segments
7. if a small touched batch fits the patch-tier bound, writes the batch as patch-tier segments without rewriting old data
8. otherwise normalizes by merging the batch, all live patch segments, and the intersecting main-tier segments into one sorted, deduplicated batch (winner rule for duplicate keys)
9. splits the direct, patch, or normalized batch by the configured flush thresholds and writes immutable temp segments
10. renames completed segments into `segments/`
11. atomically publishes a replacing `MANIFEST` that removes normalized or dead entries and inserts the new segments
12. deletes the retired segment files best-effort

Tail appends, gap inserts, and interleaving batches are all allowed: an interleaving batch first enters the patch tier when it fits, then normalization rebuilds the touched region when the patch tier reaches its bound. See [Steady-State Write Design](#steady-state-write-design).

There is no WAL. Only fully published segments are durable. In-progress temp segments may be lost on crash. A crash after segment rename but before manifest publication leaves an orphan final segment, which is invisible and is collected by GC.

## Corruption Semantics

This store follows cache semantics:

- corrupted header: whole segment ignored
- corrupted footer or block index: whole segment ignored
- block checksum failure: that block behaves like missing data
- malformed block: that block behaves like missing data
- orphan temp file: ignored
- orphan final segment not referenced by manifest: ignored
- missing segment referenced by manifest: behaves like missing keyspace
- referenced segment with mismatched header/footer metadata: ignored

The store is allowed to forget data. It is not allowed to return wrong data.

Manifest entries whose segment files fail to open are dead entries: invisible at open time and dropped at the next manifest publication, so lost ranges become writable miss space instead of staying reserved. Disagreement between two copies of one key is not corruption; it is expected under the caller contract and resolved by the winner rule in [Duplicate Keys and Convergence](#duplicate-keys-and-convergence).

## Process Model

Within one process:

- `Store` is a cheaply cloneable shared handle.
- Readers take an atomic snapshot of the visible segment set; commits serialize on an internal lock and never block readers on file I/O.
- Lookup sessions and scan cursors hold immutable segment state, so they remain valid across commits; they observe the snapshot taken at the start of each call.

Across processes:

- Concurrent read-only opens of one root are safe: segments are immutable and the manifest swap is atomic. On Unix, an open file descriptor remains readable after the file is unlinked by GC.
- A reader's view is the manifest snapshot read at open time; reopen to observe later publications.
- Concurrent writable opens are rejected by the advisory writer lock. The lock is cooperative; external tools that mutate the root without taking it can still corrupt visibility.

A writer takes an advisory lock on `LOCK` (`std::fs::File::try_lock`, which is `flock` on Unix) and holds it for the lifetime of the `Store` handle, covering creation, open-time GC, commits, and post-commit GC. The lock file is stable and is never replaced by atomic publication, so two writers cannot accidentally lock different inodes. A second writer fails fast (`InputError::WriterLocked`) instead of corrupting visibility. Read-only opens (`OpenOptions::with_read_only`) do not take the lock, run no GC, and never mutate the filesystem; commits on a read-only handle are rejected with `InputError::ReadOnlyStore`. Sync agents will count as writers.

## Steady-State Write Design

This section addresses why insert-only commits cannot sustain the intended workload: demand-driven cache filling produces batches that interleave with published ranges, and segment count grows without bound. Replacing manifest commits, commit routing, normalization, and garbage collection are implemented.

### Replacing Manifest Commit

The original commit path could only insert segments into gaps. The generalized primitive is:

> One manifest publication atomically removes a set of existing segments and inserts a set of new segments.

The `MANIFEST` snapshot mechanism already supports this; only the validation layer forbids it. A replacing commit must leave the main tier sorted and non-overlapping, and every key in a removed segment must either be re-published by an inserted segment or belong to a dropped dead entry.

This primitive provides local rebuild (merge the intersecting segments with the batch and publish replacements), segment-count convergence (coalesce adjacent undersized segments while already rewriting a region), and dead-entry removal (drop entries whose files failed to open).

Crash semantics are unchanged: new segments are invisible orphans until the manifest publication, which remains the single atomicity point.

A replacing commit is demand-driven and local. There is no background compaction: a segment is rewritten only when a commit interleaves with its range, so regions that stop receiving writes become permanently stable files. This is a permanent invariant, load-bearing for sync, not an optimization.

### Commit Routing

A commit chooses one of three routes per batch:

1. **Direct to main tier** when the batch does not overlap any main-tier segment. This is the preferred route for bulk publishes and gap inserts.
2. **Patch tier** when the batch overlaps main-tier data but still fits the patch policy. The batch is published as one or more patch segments, rewriting nothing.
3. **Normalization** when the patch policy would be exceeded. The batch, all live patch segments, and the intersecting main-tier segments are merged into replacement main-tier segments.

The current patch policy is intentionally simple: at most 8 live patch segments, and only batches with at most 4096 input records publish directly to patch. When a publication would exceed the bound, the commit performs normalization: merge all patch segments plus the intersecting main-tier segments into new main-tier segments, deduplicating by the winner rule and dropping dead entries.

The bound is a normalization trigger. Read amplification is bounded by the live patch count plus the main stream in steady state.

Write amplification is bounded at roughly one patch write plus one normalization merge per byte, comparable to a two-level LSM floor. Read amplification is bounded by `K` per root; deployments with many roots must budget `active roots x K` in aggregate.

### Garbage Collection

- After every manifest publication, a full GC pass runs: every file in `segments/` that the just-published manifest does not reference is deleted best-effort. A long-running writer therefore reclaims retired segments and orphans from earlier failed publications without a reopen.
- On open, under the writer lock: the same pass runs against the loaded manifest, plus stale catalog temp files (`STORE.tmp`, `MANIFEST.tmp`) in the root are removed. Unreferenced files are provably dead because the manifest is the sole source of visibility and commits never adopt pre-existing files.
- GC, commits, and sync ingestion all run under the writer lock, so staged remote files cannot be collected mid-ingest: a sync agent places files and publishes the manifest that references them without releasing the lock in between.

GC also removes the v1 failure mode where an orphan file left by a crash between segment rename and manifest publication permanently occupies its segment id and blocks all future commits.

GC's directory scan exists only to delete dead files. Discovery of visible data still never depends on directory contents; `MANIFEST` remains the sole visibility source. File deletion is best-effort everywhere: on platforms where open files cannot be unlinked, failed deletions are simply retried by a later GC pass.

## Duplicate Keys and Convergence

The winner rule below is implemented: a replacing commit that merges a key present in both the batch and an existing segment keeps the lexicographically smallest value. Simultaneous copies of one key across visible segments arise while patch-tier segments are live; normalization leaves the main tier deduplicated.

The general invariant, which the patch tier relies on:

> Copies of one key may exist across visible segments. All copies are semantically interchangeable (caller contract); their bytes may differ.

Byte differences between copies are expected, not exceptional: floating-point library results differ across platforms, and codec or compression versions differ across builds. The store never compares value bytes to detect errors and never treats copy disagreement as corruption.

- **Reads** return the lexicographically smallest value bytes among visible copies.
- **Normalization** deduplicates. The surviving copy is the **winner**: the copy with the lexicographically smallest value bytes. The winner is a function of the copies alone - not of arrival order, merge history, or wall-clock time - so deduplication is commutative and associative, and replicas converge to the same survivor regardless of the order in which they sync and normalize. (A "keep oldest" rule was considered and rejected: "oldest" is local history, and history-dependent winners prevent replicas from ever converging byte-for-byte.)

Convergence property, which sync relies on: two roots that hold the same key set and use the same writer configuration converge to byte-identical segment sets once both are fully normalized. This follows from deterministic segment encoding, the deterministic winner rule, and deterministic merge split points.

## Cross-Device Sync

**Decided.** Immutable segments are the sync unit; cold segments never change, so file-level incremental sync stays effective indefinitely. Sync reuses the write-path primitives - a remote segment is ingested like a local publication, and normalization is an ordinary replacing commit - rather than adding a parallel mechanism.

The transport is out of scope: the protocol assumes only a file-copy mechanism (rsync, object storage, manual copy) and read access to the remote `MANIFEST` snapshot.

### Prerequisites

- Both roots must have byte-identical `STORE` identity: metadata, `key_len`, `value_len`. `STORE` itself is never synced; sync between mismatched roots is refused.
- A new replica is bootstrapped by creating an empty store with identical creation options (or by copying `STORE` before the first ingestion). After bootstrap, `STORE` is never written again.
- Cross-device key stability is required (see [Caller Contract](#caller-contract)).
- Sync agents are writers and hold the advisory writer lock during ingestion.

### Content-Addressed Segment Identity

Segment files are named by the BLAKE3-256 hash of their bytes, and manifest entries reference that hash. This replaces the `next_segment_id` counter, which is a single-writer construct: independent writers would allocate colliding ids for different content.

Consequences:

- file-name collisions between devices disappear; identical files are the same file
- identical work performed on two devices deduplicates automatically - but only when entries and writer configuration match exactly, so dedup is an opportunistic optimization that correctness never depends on
- sync verification upgrades from per-block CRC32C to an end-to-end content hash per file

The hash is computed exactly once, streamed while the segment is written. It is verified only when a file crosses a trust boundary: sync ingestion re-hashes received files against their names. The read path never verifies whole-file hashes - runtime corruption detection remains per-block CRC32C - and open does not verify them either, since open reads only the header and footer.

### One-Way Replication

The simple mode, conceptually available already:

1. Copy every segment file referenced by the source manifest into the replica's `segments/`.
2. Atomically publish the source manifest snapshot as the replica's `MANIFEST`.
3. Replica readers reopen to observe the new snapshot.

The replica must either be read-only or perform step 2 as an ordinary ingestion under its own writer lock.

### Bidirectional Sync

Reconciliation is a set union, requiring no conflict resolution:

1. Copy remote segment files that are not already present (content addressing makes presence checks trivial).
2. Under the local writer lock, publish a manifest that adds the remote entries: remote segments that do not overlap the local main tier may enter the main tier directly; overlapping ones enter the patch tier.
3. Later normalizations merge and deduplicate as usual. Duplicate copies introduced by both devices computing the same key are resolved by the winner rule.

The worst case is two devices that computed fully interleaved key ranges: ingestion lands everything in the patch tier and the next normalization rewrites the whole overlapped range once. This is the bounded price of merging two divergent histories, not a recurring cost.

### Generations and Races

Every manifest publication increments a per-root `generation` counter. Generations totally order the snapshots of one root - useful for cheap change detection and incremental pull - and are meaningless across roots; no cross-device ordering exists or is needed.

A puller that copied a manifest and then finds a referenced segment file already deleted (the source normalized in between) restarts from the source's current manifest. This wastes bandwidth, never correctness: every manifest snapshot is self-consistent, and activation happens only after all referenced files are present.

## Catalog Revision

**Decided.** The next manifest revision carries the fields required by the design above. Field list, not final byte layout; the byte layout is fixed when implemented, and existing stores are rebuilt (experimental status, no migration):

- `magic`, `version`
- `key_len` (cross-check against `STORE`)
- `generation: u64`
- `segment_count`
- per entry: `segment_ref` (BLAKE3-256 hash), `tier` (main or patch), `min_key`, `max_key`
- `crc32c` trailer

Canonical entry order: main-tier entries sorted by `min_key`, then patch-tier entries sorted by `(min_key, segment_ref)`. `next_segment_id` is retired with content addressing.

## Current Implementation Optimizations

### Sparse In-Memory Block Index

Each segment loads one sparse block index entry per block into memory. This keeps segment-open cost modest while making block selection cheap.

### Borrowed Block Parsing

Blocks are stored in memory as raw bytes plus layout metadata. The reader parses records by reference and only copies the value when it must return ownership.

For fixed-value blocks, the reader does not need a value offset table; it computes the value slice from the record index and configured value length.

### Batch Block Consumption

Ordered lookup processes one block and consumes all keys that fall within that block range before advancing. This avoids reloading or re-searching the same block for every key in a locality-heavy ordered stream.

### Position-Independent Block Reads

Segment files are read using offset-based reads rather than mutating a shared file cursor. Lookup sessions and scan cursors also reuse their previous block buffer, so steady-state ordered reads of same-sized blocks avoid an extra zero-fill before each disk read.

### Oversized-Block Space Control

Small blocks still align to the target block size. Large blocks are written at natural size instead of being rounded up. This greatly reduces space amplification for large values without changing the read contract.

### Fixed-Value Block Layout

Fixed-value namespaces can opt into `CreateOptions::with_fixed_value_len(value_len: NonZeroU32)`. This keeps the same ordered lookup contract but removes two `u32` tables per record from the physical block. For small fixed-size values, this reduces block size and a small amount of lookup CPU. It is an explicit store-level choice rather than an automatic per-segment inference, so all visible data in one store has one stable value layout.

## Implementation Sequencing

1. **Hygiene first** — *implemented*: creation-order fix, open-time GC, advisory writer lock. They remove both permanent-wedge failure modes (orphan segment id collision, half-created root).
2. **Replacing manifest commit**, including dead-entry dropping — *implemented*. This fixes interleaved writes, segment-count growth, and reserved dead ranges.
3. **Patch tier and normalization routing** — *implemented*. This reduces rewrite amplification for repeated interleaving commits while keeping the main tier sorted and non-overlapping.
4. **Content addressing, generations, and the sync protocol** — *pending*, last, together with the catalog revision. Format changes rebuild existing stores.
