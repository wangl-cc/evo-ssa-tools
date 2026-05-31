#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
#![cfg_attr(coverage_nightly, allow(unused_features))]

//! Stable schema fingerprints for cache wire formats.
//!
//! `CacheSchema` describes the serialized shape of a type, not its Rust memory layout. The
//! resulting canonical bytes are hashed with BLAKE3 and truncated to 128 bits.
//!
//! ```rust
//! use ssa_cache_schema::{CacheSchema, schema_fingerprint};
//!
//! #[derive(CacheSchema)]
//! struct Params {
//!     width: u32,
//!     height: u32,
//! }
//!
//! let first = schema_fingerprint::<Params>();
//! let second = schema_fingerprint::<Params>();
//! assert_eq!(first, second);
//! ```
//!
//! Serde attributes are intentionally not interpreted by the derive macro:
//!
//! ```compile_fail
//! use ssa_cache_schema::CacheSchema;
//!
//! #[derive(CacheSchema)]
//! struct Bad {
//!     #[serde(skip)]
//!     value: u32,
//! }
//! ```
//!
//! Unsupported `cache_schema` attributes are rejected instead of being ignored:
//!
//! ```compile_fail
//! use ssa_cache_schema::CacheSchema;
//!
//! #[derive(CacheSchema)]
//! struct Bad {
//!     #[cache_schema(skip)]
//!     value: u32,
//! }
//! ```

use std::marker::PhantomData;

#[cfg(feature = "derive")]
pub use ssa_cache_schema_derive::CacheSchema;

/// A 128-bit schema fingerprint.
pub type SchemaFingerprint = [u8; 16];

/// A type that can describe its cache wire schema.
pub trait CacheSchema {
    /// Write this type's canonical schema description.
    fn write_schema(w: &mut SchemaWriter);
}

/// Compute the BLAKE3-128 schema fingerprint for `T`.
pub fn schema_fingerprint<T: CacheSchema>() -> SchemaFingerprint {
    let mut writer = SchemaWriter::new();
    T::write_schema(&mut writer);
    writer.finish_fingerprint()
}

/// Canonical writer used by [`CacheSchema`] implementations.
///
/// Every token is encoded as a one-byte tag followed by fixed-width integers or length-prefixed
/// byte strings. This keeps different schema trees from colliding through ambiguous concatenation.
#[derive(Clone)]
pub struct SchemaWriter {
    hasher: blake3::Hasher,
}

impl SchemaWriter {
    /// Create an empty schema writer.
    pub fn new() -> Self {
        Self {
            hasher: blake3::Hasher::new(),
        }
    }

    /// Write a primitive type name.
    pub fn primitive(&mut self, name: &'static str) {
        self.tagged_str(Tag::Primitive, name);
    }

    /// Begin a struct schema.
    pub fn struct_begin(&mut self, module_path: &'static str, name: &'static str) {
        self.tag(Tag::StructBegin);
        self.str(module_path);
        self.str(name);
    }

    /// End a struct schema.
    pub fn struct_end(&mut self) {
        self.tag(Tag::StructEnd);
    }

    /// Begin an enum schema.
    pub fn enum_begin(&mut self, module_path: &'static str, name: &'static str) {
        self.tag(Tag::EnumBegin);
        self.str(module_path);
        self.str(name);
    }

    /// End an enum schema.
    pub fn enum_end(&mut self) {
        self.tag(Tag::EnumEnd);
    }

    /// Write an explicit type schema version salt.
    pub fn type_version(&mut self, version: &'static str) {
        self.tagged_str(Tag::TypeVersion, version);
    }

    /// Begin a field schema.
    pub fn field_begin(&mut self, index: usize, name: Option<&'static str>) {
        self.tag(Tag::FieldBegin);
        self.usize(index);
        self.option_str(name);
    }

    /// End a field schema.
    pub fn field_end(&mut self) {
        self.tag(Tag::FieldEnd);
    }

    /// Begin an enum variant schema.
    pub fn variant_begin(&mut self, index: usize, name: &'static str) {
        self.tag(Tag::VariantBegin);
        self.usize(index);
        self.str(name);
    }

    /// End an enum variant schema.
    pub fn variant_end(&mut self) {
        self.tag(Tag::VariantEnd);
    }

    /// Begin a tuple schema.
    pub fn tuple_begin(&mut self) {
        self.tag(Tag::TupleBegin);
    }

    /// End a tuple schema.
    pub fn tuple_end(&mut self) {
        self.tag(Tag::TupleEnd);
    }

    /// Begin a sequence-like schema.
    pub fn seq_begin(&mut self, name: &'static str) {
        self.tagged_str(Tag::SeqBegin, name);
    }

    /// End a sequence-like schema.
    pub fn seq_end(&mut self) {
        self.tag(Tag::SeqEnd);
    }

    /// Begin an array schema with fixed length.
    pub fn array_begin(&mut self, len: usize) {
        self.tag(Tag::ArrayBegin);
        self.usize(len);
    }

    /// End an array schema.
    pub fn array_end(&mut self) {
        self.tag(Tag::ArrayEnd);
    }

    /// Begin a map-like schema.
    pub fn map_begin(&mut self, name: &'static str) {
        self.tagged_str(Tag::MapBegin, name);
    }

    /// End a map-like schema.
    pub fn map_end(&mut self) {
        self.tag(Tag::MapEnd);
    }

    fn tagged_str(&mut self, tag: Tag, value: &'static str) {
        self.tag(tag);
        self.str(value);
    }

    fn option_str(&mut self, value: Option<&'static str>) {
        match value {
            Some(value) => {
                self.tag(Tag::Some);
                self.str(value);
            }
            None => self.tag(Tag::None),
        }
    }

    fn str(&mut self, value: &str) {
        self.usize(value.len());
        self.write(value.as_bytes());
    }

    fn usize(&mut self, value: usize) {
        self.write(&(value as u64).to_le_bytes());
    }

    fn tag(&mut self, tag: Tag) {
        self.write(&[tag as u8]);
    }

    fn write(&mut self, bytes: &[u8]) {
        self.hasher.update(bytes);
    }

    fn finish_fingerprint(self) -> SchemaFingerprint {
        let full = self.hasher.finalize();
        full.as_bytes()[..16]
            .try_into()
            .expect("BLAKE3 hash is at least 16 bytes")
    }
}

impl Default for SchemaWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for SchemaWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SchemaWriter").finish_non_exhaustive()
    }
}

#[repr(u8)]
#[derive(Clone, Copy)]
enum Tag {
    Primitive = 1,
    StructBegin = 2,
    StructEnd = 3,
    EnumBegin = 4,
    EnumEnd = 5,
    TypeVersion = 6,
    FieldBegin = 7,
    FieldEnd = 8,
    VariantBegin = 9,
    VariantEnd = 10,
    TupleBegin = 11,
    TupleEnd = 12,
    SeqBegin = 13,
    SeqEnd = 14,
    MapBegin = 15,
    MapEnd = 16,
    Some = 17,
    None = 18,
    ArrayBegin = 19,
    ArrayEnd = 20,
}

macro_rules! impl_primitive_schema {
    ($($ty:ty => $name:literal),+ $(,)?) => {
        $(
            impl CacheSchema for $ty {
                fn write_schema(w: &mut SchemaWriter) {
                    w.primitive($name);
                }
            }
        )+
    };
}

impl_primitive_schema!(
    bool => "bool",
    char => "char",
    u8 => "u8",
    u16 => "u16",
    u32 => "u32",
    u64 => "u64",
    u128 => "u128",
    usize => "usize",
    i8 => "i8",
    i16 => "i16",
    i32 => "i32",
    i64 => "i64",
    i128 => "i128",
    isize => "isize",
    f32 => "f32",
    f64 => "f64",
);

impl CacheSchema for () {
    fn write_schema(w: &mut SchemaWriter) {
        w.tuple_begin();
        w.tuple_end();
    }
}

impl CacheSchema for String {
    fn write_schema(w: &mut SchemaWriter) {
        w.seq_begin("StringUtf8");
        u8::write_schema(w);
        w.seq_end();
    }
}

impl<T: CacheSchema> CacheSchema for Vec<T> {
    fn write_schema(w: &mut SchemaWriter) {
        w.seq_begin("Vec");
        T::write_schema(w);
        w.seq_end();
    }
}

impl<T: CacheSchema> CacheSchema for Option<T> {
    fn write_schema(w: &mut SchemaWriter) {
        w.enum_begin("core::option", "Option");
        w.variant_begin(0, "None");
        w.variant_end();
        w.variant_begin(1, "Some");
        w.field_begin(0, None);
        T::write_schema(w);
        w.field_end();
        w.variant_end();
        w.enum_end();
    }
}

impl<T: CacheSchema, E: CacheSchema> CacheSchema for Result<T, E> {
    fn write_schema(w: &mut SchemaWriter) {
        w.enum_begin("core::result", "Result");
        w.variant_begin(0, "Ok");
        w.field_begin(0, None);
        T::write_schema(w);
        w.field_end();
        w.variant_end();
        w.variant_begin(1, "Err");
        w.field_begin(0, None);
        E::write_schema(w);
        w.field_end();
        w.variant_end();
        w.enum_end();
    }
}

impl<T: CacheSchema, const N: usize> CacheSchema for [T; N] {
    fn write_schema(w: &mut SchemaWriter) {
        w.array_begin(N);
        T::write_schema(w);
        w.array_end();
    }
}

impl<T: CacheSchema + ?Sized> CacheSchema for Box<T> {
    fn write_schema(w: &mut SchemaWriter) {
        T::write_schema(w);
    }
}

impl<T> CacheSchema for PhantomData<T> {
    fn write_schema(w: &mut SchemaWriter) {
        w.primitive("PhantomData");
    }
}

macro_rules! impl_tuple_schema {
    ($($T:ident $idx:tt),+ $(,)?) => {
        impl<$($T: CacheSchema),+> CacheSchema for ($($T,)+) {
            fn write_schema(w: &mut SchemaWriter) {
                w.tuple_begin();
                $(
                    w.field_begin($idx, None);
                    $T::write_schema(w);
                    w.field_end();
                )+
                w.tuple_end();
            }
        }
    };
}

impl_tuple_schema!(T0 0);
impl_tuple_schema!(T0 0, T1 1);
impl_tuple_schema!(T0 0, T1 1, T2 2);
impl_tuple_schema!(T0 0, T1 1, T2 2, T3 3);
impl_tuple_schema!(T0 0, T1 1, T2 2, T3 3, T4 4);
impl_tuple_schema!(T0 0, T1 1, T2 2, T3 3, T4 4, T5 5);
impl_tuple_schema!(T0 0, T1 1, T2 2, T3 3, T4 4, T5 5, T6 6);
impl_tuple_schema!(T0 0, T1 1, T2 2, T3 3, T4 4, T5 5, T6 6, T7 7);
impl_tuple_schema!(T0 0, T1 1, T2 2, T3 3, T4 4, T5 5, T6 6, T7 7, T8 8);
impl_tuple_schema!(T0 0, T1 1, T2 2, T3 3, T4 4, T5 5, T6 6, T7 7, T8 8, T9 9);
impl_tuple_schema!(T0 0, T1 1, T2 2, T3 3, T4 4, T5 5, T6 6, T7 7, T8 8, T9 9, T10 10);
impl_tuple_schema!(T0 0, T1 1, T2 2, T3 3, T4 4, T5 5, T6 6, T7 7, T8 8, T9 9, T10 10, T11 11);

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn fingerprint_is_deterministic() {
        assert_eq!(
            schema_fingerprint::<(u32, bool)>(),
            schema_fingerprint::<(u32, bool)>()
        );
    }

    #[test]
    fn writer_uses_length_prefixes_for_strings() {
        let mut first = SchemaWriter::new();
        first.seq_begin("ab");
        first.primitive("c");
        first.seq_end();
        let first = first.finish_fingerprint();

        let mut second = SchemaWriter::new();
        second.seq_begin("a");
        second.primitive("bc");
        second.seq_end();
        let second = second.finish_fingerprint();

        assert_ne!(first, second);
    }

    #[test]
    fn type_alias_uses_aliased_type_schema() {
        type MyU64 = u64;

        assert_eq!(schema_fingerprint::<MyU64>(), schema_fingerprint::<u64>());
    }

    #[test]
    fn generic_containers_include_inner_type() {
        assert_ne!(
            schema_fingerprint::<Vec<u32>>(),
            schema_fingerprint::<Vec<u64>>()
        );
        assert_ne!(
            schema_fingerprint::<Option<u32>>(),
            schema_fingerprint::<Option<u64>>()
        );
        assert_ne!(
            schema_fingerprint::<Result<u32, u8>>(),
            schema_fingerprint::<Result<u64, u8>>()
        );
        assert_ne!(
            schema_fingerprint::<Result<u32, u8>>(),
            schema_fingerprint::<Result<u32, u16>>()
        );
        assert_ne!(
            schema_fingerprint::<Box<u32>>(),
            schema_fingerprint::<Box<u64>>()
        );
    }

    #[test]
    fn ownership_wrappers_are_schema_transparent() {
        assert_eq!(
            schema_fingerprint::<Box<u32>>(),
            schema_fingerprint::<u32>()
        );
        assert_eq!(
            schema_fingerprint::<Box<Box<u32>>>(),
            schema_fingerprint::<u32>()
        );
    }

    #[test]
    fn array_length_changes_fingerprint() {
        assert_ne!(
            schema_fingerprint::<[u32; 2]>(),
            schema_fingerprint::<[u32; 3]>()
        );
    }

    #[test]
    fn tuple_order_and_arity_change_fingerprint() {
        assert_ne!(
            schema_fingerprint::<(u32, u64)>(),
            schema_fingerprint::<(u64, u32)>()
        );
        assert_ne!(
            schema_fingerprint::<(u32,)>(),
            schema_fingerprint::<(u32, u32)>()
        );
    }

    #[test]
    fn string_schema_is_distinct_from_raw_byte_vec() {
        assert_ne!(
            schema_fingerprint::<String>(),
            schema_fingerprint::<Vec<u8>>()
        );
    }

    #[test]
    fn phantom_data_ignores_type_parameter() {
        assert_eq!(
            schema_fingerprint::<PhantomData<u32>>(),
            schema_fingerprint::<PhantomData<u64>>()
        );
    }

    #[test]
    fn writer_token_boundaries_are_unambiguous() {
        let mut first = SchemaWriter::new();
        first.primitive("a");
        first.primitive("bc");
        let first = first.finish_fingerprint();

        let mut second = SchemaWriter::new();
        second.primitive("ab");
        second.primitive("c");
        let second = second.finish_fingerprint();

        assert_ne!(first, second);
    }

    #[test]
    fn unit_schema_is_distinct_from_single_empty_tuple_like_field() {
        assert_ne!(schema_fingerprint::<()>(), schema_fingerprint::<((),)>());
    }

    #[test]
    fn map_schema_tokens_are_part_of_fingerprint() {
        let mut map = SchemaWriter::new();
        map.map_begin("Map");
        u32::write_schema(&mut map);
        u64::write_schema(&mut map);
        map.map_end();
        let map = map.finish_fingerprint();

        let mut seq = SchemaWriter::new();
        seq.seq_begin("Map");
        u32::write_schema(&mut seq);
        u64::write_schema(&mut seq);
        seq.seq_end();
        let seq = seq.finish_fingerprint();

        assert_ne!(map, seq);
    }

    #[test]
    fn default_writer_matches_new_writer() {
        let mut from_default = SchemaWriter::default();
        from_default.primitive("u32");

        let mut from_new = SchemaWriter::new();
        from_new.primitive("u32");

        assert_eq!(
            from_default.finish_fingerprint(),
            from_new.finish_fingerprint()
        );
    }

    #[test]
    fn debug_output_names_writer_without_exposing_hasher_state() {
        assert_eq!(format!("{:?}", SchemaWriter::new()), "SchemaWriter { .. }");
    }
}
