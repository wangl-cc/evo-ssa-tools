# SSA Cache Schema Derive

`ssa-cache-schema-derive` provides the `#[derive(CacheSchema)]` procedural macro used by the `ssa-cache-schema` runtime crate. Most users should depend on `ssa-cache-schema` with its default `derive` feature instead of depending on this crate directly.

## Supported Items

The derive macro supports named structs, tuple structs, unit structs, and enums. Field types are emitted as `<FieldTy as CacheSchema>::write_schema(w)`, so Rust type resolution determines the schema dependency rather than hashing source tokens. Zero-field unit, tuple, and named product forms are fingerprinted distinctly.

Derived enum schemas use variant order, schema names, and fields. Explicit Rust discriminants and `repr` attributes are ignored; use `#[cache_schema(version = "...")]` or a manual implementation when discriminants are part of the cache wire format.

Generic implementations add `CacheSchema` bounds for the field types that are actually written. This allows marker-only parameters such as `PhantomData<T>` without requiring `T: CacheSchema`.

Recursive schemas are unsupported in this first version. Direct structural recursion generally fails trait resolution during derive, while cycles hidden behind manual implementations can recurse when a fingerprint is computed. Recursive domains need a manual implementation that emits a non-recursive reference identity or a future definition/reference scheme.

## Attributes

The macro accepts `#[cache_schema(rename = "...")]` on fields, variants, and types to keep schema names stable across Rust renames. Rust module paths are not included, so module moves are schema-transparent. Types also support `#[cache_schema(version = "...")]` for explicit schema salt and `#[cache_schema(crate = path)]` when the runtime crate is imported through a non-default path.

Unsupported `cache_schema` keys, duplicate keys, and unions are rejected at compile time. Serde attributes are ignored rather than interpreted: `#[serde(skip)]` does not remove a field from the cache schema, and `#[serde(rename = "...")]` does not rename it for fingerprinting. `CacheSchema` is intended to describe the cache wire schema directly.
