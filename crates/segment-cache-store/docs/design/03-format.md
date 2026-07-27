# Format

This chapter specifies durable bytes and their target logical interpretation. Per-path behavior is in [04 Operations](04-operations.md).

## Directory And Publication Layout

```text
root/
  STORE
  MANIFEST
  LOCK
  segments/
    segment-0000000000.seg
```

Temporary sibling files use a `.tmp` suffix. Writers fsync a temporary file, rename it to its final name, and fsync the parent directory. Segments are visible only after a later atomic MANIFEST publication references them. Orphan and retired final segments are invisible and are removed only by explicit garbage collection. Garbage collection stops at the first deletion error; files deleted before that error remain deleted, while MANIFEST visibility is unchanged.

`STORE` is the creation-completion marker and immutable namespace identity. It contains caller metadata encoded as lowercase hex. `MANIFEST` is an XXH3-128-protected binary snapshot that owns `key_len`, `value_len` (`0` means variable), `next_segment_id`, and visible segment references. `SegmentContentId` is `BLAKE3("scs-segment-v1\\0" || complete_segment_bytes)`; it is content identity, not an authentication tag.

Creation writes MANIFEST first and STORE last. If creation stops before STORE appears, open rejects the incomplete root and creation may safely recreate it. A completed root has both files. STORE is line-oriented text rather than JSON:

```text
segment-cache-store store v1
version=1
metadata=<hex>
key_len=<u32>
value_len=<u32> # 0 = variable
value_payload_compression_id=<u8>
```

The target MANIFEST envelope is grouped by ownership:

```text
magic[4]              # b"SCSM"
version:u32           # 1
key_len:u32
value_len:u32
next_segment_id:u32
component_count:u32
components[component_count]:
  main: SegmentRef
  patch_count:u32
  patches[patch_count]: SegmentRef
manifest_checksum[16] # canonical big-endian unseeded XXH3-128 over all preceding bytes
```

Each `SegmentRef` carries `segment_id:u32`, `segment_len:u64`, `segment_content_id[32]`, `min_key[key_len]`, and `max_key[key_len]`. Tier and Patch ownership come from the enclosing component, so they are not duplicated as per-reference fields. A segment filename is `segments/segment-<id>.seg`; filename order does not define key order. Declared length supports cheap truncation and bounds checks.

## Target Manifest Semantics

The wire form and canonical logical form both group each Main with its owned PatchSet:

```text
MainComponent {
  main: SegmentRef,
  patches: [SegmentRef ordered by min_key],
}
```

Every referenced range has valid fixed-width inclusive bounds. Components are sorted by Main minimum key, and Main ranges are globally nonoverlapping. Each Patch range is contained in its enclosing Main range; patches of that owner are sorted and nonoverlapping. The parser checks the bounded file length and complete MANIFEST checksum before trusting header counts or allocating entries, then rejects duplicate or reused segment IDs, duplicate content references, invalid ranges, noncanonical order, and geometry that exceeds format limits. It validates the hard Main cap and per-component PatchSet file and byte budgets before constructing the trusted grouped representation. The whole-file checksum already covers the MANIFEST header, so the unchunked format has no separate header checksum.

The numeric Main cap and PatchSet file and byte budgets remain TODO decisions. They are required policy limits, not wire integer-width limits.

> Current implementation note: the current implementation stores a flat sequence of all Main entries followed by all Patch entries, ordered by `(min_key, segment_id)`, and validates one patch in an interval-connected component. Because the format is still pre-release, the grouped representation will replace those experimental bytes while retaining target format version `1`; there is no migration path.

## Segment Format

Each immutable segment contains `SegmentHeaderV1`, one or more data blocks, and `SegmentFooterV1`. Empty record streams do not publish a segment. Multi-byte integers are little-endian. The first data block follows the fixed 18-byte header immediately.

```text
magic[4]             # b"SCSG"
format_version:u8    # 1
value_payload_compression_id:u8
key_len:u32
value_len:u32        # 0 = variable; otherwise fixed value length
header_crc32c:u32    # CRC32C over the previous 14 bytes
```

The header is self-describing. Its schema must match MANIFEST, and its compression ID must be supported by the reader. Format version 1 itself fixes canonical big-endian unseeded XXH3-128 for the Footer and block sections; there is no independent checksum ID. After validation, downstream decoding uses the header's own compression declaration rather than an external expected compression setting.

The footer owns the sparse block index and is the completion marker:

```text
segment_prefix_len:u32
segment_prefix[segment_prefix_len]
block_count:u32
block_index_entries[block_count]
footer_body_len:u32
footer_checksum[16]  # canonical big-endian XXH3-128 over footer body | footer_body_len
```

The Footer is one bounded, unchunked structure and is always read in full, so it has one checksum and no separate header checksum. Each index entry is:

```text
extra_prefix_len:u32
extra_prefix[extra_prefix_len]
min_suffix[key_len - segment_prefix_len - extra_prefix_len]
max_suffix[key_len - segment_prefix_len - extra_prefix_len]
block_offset:u64
```

`segment_prefix` is the maximal common prefix of the manifest segment bounds, and `extra_prefix` is the additional maximal prefix shared by a block's relative bounds. Segment-level minimum and maximum keys are not repeated in the footer. Block lengths derive from adjacent offsets; the final block ends at the footer offset obtained from the trailer. Opening validates prefix canonicality, strict block-range ordering and nonoverlap, manifest-range agreement, and every offset boundary before materializing byte ranges.

Each decoded block must agree with its validated index entry's prefix and exact minimum and maximum suffixes. A header, footer, or index failure makes the whole segment absent on an ordinary path; disagreement or corruption confined to one block makes only that block absent. Exact index bounds route gaps without loading a candidate block, and the runtime index shares footer prefix backing rather than allocating complete bounds per block.

Segment encoding is deterministic for the same canonical record stream and writer options. This supports content-addressed remote reuse and makes format changes reviewable.

## Block Format

Blocks have no envelope: their byte ranges and key ranges come from the footer index. `record_count` is the first field in checksummed lookup metadata and determines the remaining geometry. Lookup metadata and the value-payload frame each end in their own canonical big-endian XXH3-128 digest.

Variable-value blocks encode:

```text
record_count:u32
KeySection:
  extra_prefix_len:u32
  extra_prefix[extra_prefix_len]
  key_suffixes[record_count * (key_len - segment_prefix_len - extra_prefix_len)]
ValueIndexSection:
  value_offsets[record_count + 1]:u32
lookup_checksum      # over record_count | KeySection | ValueIndexSection
ValuePayloadFrame:
  optional frame header
  values...
value_checksum       # over ValuePayloadFrame
```

Fixed-value blocks encode:

```text
record_count:u32
KeySection:
  extra_prefix_len:u32
  extra_prefix[extra_prefix_len]
  key_suffixes[record_count * (key_len - segment_prefix_len - extra_prefix_len)]
lookup_checksum      # over record_count | KeySection
ValuePayloadFrame:
  optional frame header
  values[record_count * value_len]
value_checksum       # over ValuePayloadFrame
```

For variable values, `value_offsets[i]..value_offsets[i + 1]` selects value `i`, and the final offset is the decoded payload length. For fixed values, value `i` starts at `i * value_len`. Lookup metadata includes the variable-value offset table because corrupt offsets could otherwise return a wrong slice for a valid key.

There is no stored payload offset or payload length. The decoder derives:

```text
payload_frame_offset = lookup_metadata_len + 16
decoded_payload_len  = value_offsets[record_count]       # variable values
decoded_payload_len  = record_count * value_len          # fixed values
encoded_frame_len    = block_len - payload_frame_offset - 16
```

Every derivation uses checked arithmetic and must consume the complete block; v1 permits no trailing padding.

The segment header chooses the compression algorithm. `None` stores raw payload bytes without a frame header. A compression-capable segment uses:

```text
storage:u8   # 0 = raw fallback, 1 = compressed with the segment-header codec
payload[...] # extends to value_checksum
```

The block boundary gives the encoded size and the value layout gives the decoded size. Writers may use raw fallback for small or incompressible blocks. The frame does not repeat an algorithm ID, raw length, or encoded length.

Structural validation requires a nonzero `record_count`; an `extra_prefix` that fits the remaining fixed key width; strictly increasing suffixes; exact first and last keys matching the footer index; and, for variable values, an offset table that starts at zero, is monotonic, and ends at the decoded payload length. A raw frame must have exactly the derived decoded length. A compressed frame must decode to exactly that length without trailing frame bytes. These checks run even when checksum verification is disabled.

Each block strips the footer-owned segment prefix and stores only the additional common prefix shared by its first and last relative keys. Sorted order makes both prefixes common to every block key. This layout preserves raw-block views: readers load a complete block in one positioned read, validate derived ranges over that one backing buffer, binary-search fixed-width suffixes, and avoid decompression for a miss-only probe. Ordered cursors reconstruct only the current full key in reusable scratch; visitors borrow raw or decoded value slices rather than allocating each value.

## Block Sizing And Compression Policy

`target_block_size` is a logical grouping target, not a minimum physical size. Encoded sizes include metadata, optional compression, and checksums, and existing segments remain readable when later commits choose a different target. Compression policy is a write-time choice below the caller codec and above raw block payload bytes; it does not alter keys or value indexes. The v1 default attempts compression only for raw payloads of at least 64 KiB and retains it only when it saves at least 20%.

## Format Safety Limits

Wire widths are not allocation budgets. v1 bounds key length to 64 KiB, one value and decoded block payload to 64 MiB, an encoded block to 128 MiB, a footer to 64 MiB, a segment to 1,048,576 blocks, and a manifest to 256 MiB. Manifest, footer, block, compression-frame, and transfer geometry use checked arithmetic, bounded reads, complete-consumption checks, and fallible reservation before allocation or decompression.

## Current Implementation Notes

The current pre-release implementation additionally persists schema and encoding fields in `STORE`, persists only `key_len` in `MANIFEST`, compares each segment compression ID with a store-level setting, and does not yet select compression through `CommitOptions`. Rebuilding an experimental cache is the migration path when these target format semantics land.
