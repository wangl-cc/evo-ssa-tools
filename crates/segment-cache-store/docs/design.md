# Design

## Purpose

`segment-cache-store` is a storage backend for computation-cache workloads, not a general-purpose database.

The intended workload has these properties:

- keys are fixed-width byte strings
- key order is meaningful and query order follows key order
- values are already serialized and are opaque to the store
- each store declares one value layout: variable-length values or fixed-length values
- inserts are append-only
- ordered batch lookup is the hot path
- corruption should degrade to cache misses, not incorrect results

This backend intentionally avoids functionality that is expensive but not useful for this workload:

- no transactions
- no deletes
- no update-in-place
- no compaction
- no multi-writer coordination
- no WAL-based recovery

## Optimization Targets

The implementation optimizes for:

1. ordered batch lookup
2. append-only publish throughput
3. immutable file layout for sync and inspection
4. low space amplification for narrow-distribution namespaces
5. simple and explicit corruption handling

It is not optimized for:

- random write workloads
- random point lookups across many unrelated key regions
- long-term segment-count management

## Core Invariants

- one store has one fixed key length
- one store has one fixed value layout
- published segment files are immutable
- each shard owns a strictly increasing sequence of non-overlapping key ranges
- within a shard, a key can only appear once in published data
- the manifest is the source of truth for visible data
- missing or corrupted data is treated as absent, never as valid output

## Directory Layout

Each store directory contains:

```text
root/
  MANIFEST
  shards/
    0/
      segments/
        segment-00000000000000000000.seg
        ...
    1/
      segments/
        ...
  tmp/
```

The current manifest format version is `1`.

The manifest stores:

- manifest format version
- key length
- value layout
- shard count
- shard key offset
- target block size used by the most recent writer
- shard algorithm version
- next segment id
- visible segments for each shard

Segments on disk that are not referenced by the manifest are ignored.

The manifest is a line-oriented text file, not JSON. It is intentionally small and fixed-format so the crate does not need a general serialization dependency for store metadata. Binary keys are encoded as lowercase hex in segment entries.

The manifest layout is:

```text
segment-cache-store manifest v1
version=1
key_len=<usize>
value_layout=variable | fixed:<value_len>
shard_count=<usize>
shard_key_offset=<usize>
target_block_size=<usize>
shard_algorithm=lexicographic-prefix-v1
next_segment_id=<u64>
[shard 0]
segment<TAB>file_name<TAB>min_key_hex<TAB>max_key_hex<TAB>record_count<TAB>created_at_unix_millis
...
```

## Segment File Layout

Each segment file contains:

```text
data blocks
block index
footer
footer crc
footer length
magic bytes
```

All integer fields in segment files are little-endian. The current segment format version is `1`, and the trailing footer magic is `scsft001`.

The footer stores:

- segment format version
- shard id
- key length
- value layout tag
- fixed value length, or zero for variable values
- codec version
- record count
- block index offset
- block index length
- block index crc
- min key
- max key

The footer payload layout is:

```text
version: u32
shard_id: u32
key_len: u32
value_layout_tag: u32       // 0 = variable, 1 = fixed
fixed_value_len: u32        // 0 for variable layout
codec_version: u32
record_count: u64
block_index_offset: u64
block_index_len: u64
block_index_crc: u32
min_key[key_len]
max_key[key_len]
footer_crc32c: u32
footer_len: u32
magic[8]
```

The reader opens a segment by reading the footer from the end of the file, validating the footer and block index, then loading the sparse block index into memory.

If the footer or block index is invalid, the whole segment is ignored.

## Value Layout

Each store uses one `ValueLayout`:

- `Variable`: values may have different byte lengths
- `Fixed { value_len }`: every value must have exactly `value_len` bytes

The layout is stored in both the manifest and each segment footer. Opening an existing store with a different value layout is a manifest mismatch. Opening a segment whose footer layout does not match the configured store layout ignores that segment rather than attempting to decode it.

Fixed layout is intended for namespaces where the serialized output is a fixed-size struct, tuple, scalar vector, or other stable binary representation. It removes the per-record value offset and length tables from each block and lets the reader compute the value slice directly from the key index.

## Record and Block Layout

Each segment stores one codec version in the footer. If the segment codec version does not match the store's configured codec version, the whole segment is ignored. This prevents old serialized bytes from being returned under a newer value codec. Because the store is append-only and does not overwrite keys, callers that need automatic recomputation under a new codec should still include the schema/cache version in the namespace or key encoding.

All blocks start with the same 16-byte header:

```text
record_count: u32
block_len: u32
value_area_offset: u32
value_area_len: u32
...
```

Each block stores a 4-byte CRC32C trailer immediately after the logical value area and before any padding. The block checksum covers the logical block bytes before the checksum trailer. Padding bytes are not included in the checksum.

`StoreOptions::with_block_checksum_verification(false)` disables data-block checksum verification for benchmarking against engines that do not validate user value bytes on read. This is not the default cache-safe mode: with verification disabled, corrupted value bytes may be returned as hits.

### Variable-Value Blocks

Variable-value blocks store a contiguous fixed-width key table, then per-record value offsets and lengths, then the packed value bytes:

```text
record_count: u32
block_len: u32
value_area_offset: u32
value_area_len: u32
keys[record_count * key_len]
value_offsets[record_count]: u32
value_lens[record_count]: u32
values...
block_crc32c: u32
padding...
```

The key table is fixed-width and contiguous. Lookup binary-searches this key table, then uses the matching value offset and length to slice into the value area.

The value table allows variable-length values while keeping key search independent from large value bytes.

### Fixed-Value Blocks

Fixed-value blocks omit the value offset and length tables:

```text
record_count: u32
block_len: u32
value_area_offset: u32
value_area_len: u32
keys[record_count * key_len]
values[record_count * value_len]
block_crc32c: u32
padding...
```

Lookup binary-searches the key table, then computes the value slice as:

```text
value_start = value_area_offset + index * value_len
value_end   = value_start + value_len
```

The reader validates that `value_area_len == record_count * value_len`. The writer rejects any value whose length does not match the configured fixed value length, and no segment is published for that commit.

Block sizing rules:

- blocks smaller than the target block size are padded up to the target size
- oversized blocks are written at their natural size and are not padded further

This keeps small and medium namespaces readahead-friendly while avoiding severe space amplification for large values.

The target block size is a writer policy, not a read compatibility invariant. Existing segments store their physical block lengths in the block index, so a store can be reopened with a different target block size and future segments can use the new size.

## Read Path

### Point Lookup

`fetch_one` works in four stages:

1. select the shard from the key prefix
2. binary-search segments in that shard by key range
3. binary-search the segment block index
4. load the block and find the record

The point-lookup path exists for completeness, but it is not the primary optimization target.

### Ordered Batch Lookup

Ordered lookup is the main hot path.

The low-level APIs are:

- `Store::probe_ordered`
- `Store::fetch_many_ordered`
- `Store::visit_many_ordered`
- `Store::lookup_session`
- `OrderedLookup::probe_many`
- `OrderedLookup::fetch_many`
- `OrderedLookup::visit_many`

`visit_many` is the lowest-level ordered-read API. It lets callers consume borrowed value slices instead of forcing a `Vec<u8>` allocation for every hit.

The current ordered lookup implementation does not treat each key as a separate point lookup. Instead it:

1. groups consecutive keys by shard
2. walks shard-local segments in key order
3. keeps the current block loaded
4. consumes all keys that fall inside that block range before advancing

This reduces repeated block loads and makes large ordered hit streams much cheaper than independent lookups.

The store also exposes `visit_many_ordered_slice` for callers that already hold keys in a slice. This avoids the extra key-reference allocation required by the fully generic iterator API.

### Iteration

`iter_all()` returns all visible records in order.

`range(start, end)` returns the half-open range `[start, end)`.

The cursor is streaming at block granularity. It clones the visible segment handles at cursor creation, then reads one block at a time from each active segment and performs a k-way merge over the current records. It does not materialize the full scan result before returning.

If the visible segment ranges are globally non-overlapping after sorting by `min_key`, the cursor uses a concatenation fast path instead of a per-record k-way merge. This is common for prefix-preserving sharding and append-only segment ranges, and it avoids repeated cross-cursor key comparisons during full scans.

## Write Path

The public write path is:

- `Store::begin_batch`
- `WriteBatch::push`
- optional `WriteBatch::mark_sorted`
- `Store::commit_batch`

`commit_batch`:

1. validates key lengths
2. validates fixed value lengths when fixed layout is enabled
3. partitions the batch by shard
4. sorts shard-local keys if needed
5. rejects duplicate keys inside one shard batch
6. rejects shard-local overlap with already-published data
7. splits each shard-local batch by the configured flush thresholds
8. writes immutable segments for each non-empty shard-local chunk
9. atomically publishes the updated manifest

There is no WAL. Only fully published segments are durable. In-progress segments may be lost on crash.

## Corruption Semantics

This store follows cache semantics:

- block checksum failure: that block behaves like missing data
- malformed block: that block behaves like missing data
- malformed footer or block index: the whole segment is ignored
- orphan temp file: ignored
- orphan segment not referenced by manifest: ignored
- missing segment referenced by manifest: behaves like missing keyspace

The store is allowed to forget data. It is not allowed to return wrong data.

## Current Implementation Optimizations

### Prefix-Preserving Sharding

The current shard algorithm uses lexicographic prefix partitioning rather than hash partitioning.

This preserves the natural locality of ordered key streams. Hash-based sharding destroyed that locality and hurt ordered lookup performance significantly.

The prefix begins at `StoreOptions::shard_key_offset`. The default skips a leading 16-byte namespace/version prefix, which avoids putting every record in shard 0 when the first key fields are constant. Stores persist the shard key offset in the manifest because changing it would change shard assignment.

### Sparse In-Memory Block Index

Each segment loads one sparse block index entry per block into memory. This keeps segment-open cost modest while making block selection cheap.

### Borrowed Block Parsing

Blocks are stored in memory as raw bytes plus layout metadata. The reader parses records by reference and only copies the value when it must return ownership.

For fixed-value blocks, the reader does not need a value offset table; it computes the value slice from the record index and configured value length.

This matters most for:

- probe-only lookups
- `visit_many`
- large-value namespaces

### Batch Block Consumption

Ordered lookup no longer reloads or re-searches a block for every key. It processes one block and consumes all keys that fall within that block range before advancing.

### Position-Independent Block Reads

Segment files are read using offset-based reads rather than mutating a shared file cursor. Lookup sessions and scan cursors also reuse their previous block buffer, so steady-state ordered reads of same-sized blocks avoid an extra zero-fill before each disk read.

### Oversized-Block Space Control

Small blocks still align to the target block size. Large blocks are written at natural size instead of being rounded up.

This greatly reduces space amplification for large values without changing the read contract.

### Fixed-Value Block Layout

Fixed-value namespaces can opt into `StoreOptions::with_fixed_value_len(value_len)`. This keeps the same ordered lookup contract but removes two `u32` tables per record from the physical block. For small fixed-size values, this reduces block size and a small amount of lookup CPU. It is an explicit store-level choice rather than an automatic per-segment inference, so all visible data in one store has one stable value layout.
