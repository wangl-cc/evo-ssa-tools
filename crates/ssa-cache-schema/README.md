# SSA Cache Schema

`ssa-cache-schema` provides stable schema fingerprints for cache wire formats. It describes the logical shape of a type, writes that schema into a deterministic token stream, and hashes it with BLAKE3 truncated to 128 bits.

The crate is intentionally separate from `ssa-workflow` cache storage. It can be used to evolve schema fingerprinting independently before deciding how persistent caches should consume the fingerprint.

## Basic Usage

Derive `CacheSchema` for ordinary structs and enums, then call `schema_fingerprint::<T>()` when a stable schema fingerprint is needed.

```rust
use ssa_cache_schema::{schema_fingerprint, CacheSchema};

#[derive(CacheSchema)]
struct Params {
    width: u32,
    height: u32,
}

let first = schema_fingerprint::<Params>();
let second = schema_fingerprint::<Params>();
assert_eq!(first, second);
```

The default `derive` feature re-exports the derive macro from `ssa-cache-schema-derive`. Disable default features if you only need the runtime trait and want to provide manual implementations.

## Compatibility Attributes

Use `#[cache_schema(rename = "...")]` when a Rust field, variant, or type is renamed but should keep its previous schema name.

```rust
use ssa_cache_schema::CacheSchema;

#[derive(CacheSchema)]
struct Resized {
    #[cache_schema(rename = "width")]
    w: u32,
    height: u32,
}
```

Rust module paths are not included, so moving a type between modules does not change its fingerprint. Use `#[cache_schema(version = "...")]` to intentionally change the fingerprint when a semantic format version changes or when a same-shaped type should not remain cache-compatible.

Serde attributes are ignored by `CacheSchema`. For example, `#[serde(skip)]` does not remove a field from the cache schema and `#[serde(rename = "...")]` does not rename it for fingerprinting. Use `cache_schema` attributes or a manual implementation when serde behavior should affect cache compatibility.

Field reorder, field add/remove, field type changes, enum variant reorder, and enum variant add/remove change the fingerprint by default.

## Provided Standard Schemas

The runtime crate provides schemas for primitive numeric types, `bool`, `char`, `()`, tuples up to arity 12, arrays, `String`, `str`, slices, `Vec<T>`, `Option<T>`, `Result<T, E>`, `Box<T>`, references, `Cow<'_, T>`, `PhantomData<T>`, common map/set collections, and numeric wrappers in `std::num`.

Borrow and ownership wrappers are transparent: `&T`, `&mut T`, `Box<T>`, and `Cow<'_, T>` use `T`'s schema. `String` and `str` share one text schema, and `Vec<T>` and `[T]` share one sequence schema.

`HashMap<K, V>` and `BTreeMap<K, V>` share the same logical map schema. `HashSet<T>` and `BTreeSet<T>` share the same logical set schema. This schema only describes the cache value shape; deterministic value encoding still needs to handle map/set iteration order at the serialization layer.

`Wrapping<T>` and `Saturating<T>` use the same schema as their inner numeric representation. `NonZero*` integer wrappers use distinct schemas because they reject zero values; changing between `u32` and `NonZeroU32` is treated as a cache-incompatible schema change.

## Writer Contract

`SchemaWriter` streams canonical schema tokens directly into BLAKE3 rather than storing schema bytes. Each token uses an explicit tag plus fixed-width integers or length-prefixed strings, so adjacent values cannot be misread as a different schema tree.

`CacheSchema` implementations should describe the serialized cache format, not Rust memory layout. Recursive schemas are not expanded automatically in this first version; write a manual implementation or introduce an explicit reference scheme before using recursive types.
