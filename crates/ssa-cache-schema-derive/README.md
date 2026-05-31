# SSA Cache Schema Derive

`ssa-cache-schema-derive` provides the `#[derive(CacheSchema)]` procedural macro used by the `ssa-cache-schema` runtime crate. Most users should depend on `ssa-cache-schema` with its default `derive` feature instead of depending on this crate directly.

## Supported Items

The derive macro supports named structs, tuple structs, unit structs, and enums. Field types are emitted as `<FieldTy as CacheSchema>::write_schema(w)`, so Rust type resolution determines the schema dependency rather than hashing source tokens.

Generic implementations add `CacheSchema` bounds for the field types that are actually written. This allows marker-only parameters such as `PhantomData<T>` without requiring `T: CacheSchema`.

## Attributes

The macro accepts `#[cache_schema(rename = "...")]` on fields, variants, and types to keep schema identity stable across Rust renames. Types also support `#[cache_schema(module = "...")]` for module moves, `#[cache_schema(version = "...")]` for explicit schema salt, and `#[cache_schema(crate = path)]` when the runtime crate is imported through a non-default path.

Unsupported `cache_schema` keys, duplicate keys, serde attributes, and unions are rejected at compile time. The macro does not interpret serde behavior; `CacheSchema` is intended to describe the cache wire schema directly.
