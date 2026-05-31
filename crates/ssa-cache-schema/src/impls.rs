use std::marker::PhantomData;

use crate::{CacheSchema, SchemaWriter};

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
        str::write_schema(w);
    }
}

impl CacheSchema for str {
    fn write_schema(w: &mut SchemaWriter) {
        w.seq_begin("String");
        u8::write_schema(w);
        w.seq_end();
    }
}

impl<T: CacheSchema> CacheSchema for Vec<T> {
    fn write_schema(w: &mut SchemaWriter) {
        <[T]>::write_schema(w);
    }
}

impl<T: CacheSchema> CacheSchema for [T] {
    fn write_schema(w: &mut SchemaWriter) {
        w.seq_begin("Vec");
        T::write_schema(w);
        w.seq_end();
    }
}

impl<T: CacheSchema> CacheSchema for Option<T> {
    fn write_schema(w: &mut SchemaWriter) {
        w.enum_begin("Option");
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
        w.enum_begin("Result");
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

impl<T: CacheSchema + ?Sized> CacheSchema for &T {
    fn write_schema(w: &mut SchemaWriter) {
        T::write_schema(w);
    }
}

impl<T: CacheSchema + ?Sized> CacheSchema for &mut T {
    fn write_schema(w: &mut SchemaWriter) {
        T::write_schema(w);
    }
}

impl<'a, T> CacheSchema for std::borrow::Cow<'a, T>
where
    T: CacheSchema + ToOwned + ?Sized,
{
    fn write_schema(w: &mut SchemaWriter) {
        T::write_schema(w);
    }
}

impl<T> CacheSchema for PhantomData<T> {
    fn write_schema(w: &mut SchemaWriter) {
        w.primitive("PhantomData");
    }
}

impl<K: CacheSchema, V: CacheSchema, S> CacheSchema for std::collections::HashMap<K, V, S> {
    fn write_schema(w: &mut SchemaWriter) {
        write_map_schema::<K, V>(w);
    }
}

impl<K: CacheSchema, V: CacheSchema> CacheSchema for std::collections::BTreeMap<K, V> {
    fn write_schema(w: &mut SchemaWriter) {
        write_map_schema::<K, V>(w);
    }
}

impl<T: CacheSchema, S> CacheSchema for std::collections::HashSet<T, S> {
    fn write_schema(w: &mut SchemaWriter) {
        write_set_schema::<T>(w);
    }
}

impl<T: CacheSchema> CacheSchema for std::collections::BTreeSet<T> {
    fn write_schema(w: &mut SchemaWriter) {
        write_set_schema::<T>(w);
    }
}

impl<T: CacheSchema> CacheSchema for std::num::Wrapping<T> {
    fn write_schema(w: &mut SchemaWriter) {
        T::write_schema(w);
    }
}

impl<T: CacheSchema> CacheSchema for std::num::Saturating<T> {
    fn write_schema(w: &mut SchemaWriter) {
        T::write_schema(w);
    }
}

macro_rules! impl_nonzero_schema {
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

impl_nonzero_schema!(
    std::num::NonZeroU8 => "NonZeroU8",
    std::num::NonZeroU16 => "NonZeroU16",
    std::num::NonZeroU32 => "NonZeroU32",
    std::num::NonZeroU64 => "NonZeroU64",
    std::num::NonZeroU128 => "NonZeroU128",
    std::num::NonZeroUsize => "NonZeroUsize",
    std::num::NonZeroI8 => "NonZeroI8",
    std::num::NonZeroI16 => "NonZeroI16",
    std::num::NonZeroI32 => "NonZeroI32",
    std::num::NonZeroI64 => "NonZeroI64",
    std::num::NonZeroI128 => "NonZeroI128",
    std::num::NonZeroIsize => "NonZeroIsize",
);

fn write_map_schema<K: CacheSchema, V: CacheSchema>(w: &mut SchemaWriter) {
    w.map_begin("Map");
    K::write_schema(w);
    V::write_schema(w);
    w.map_end();
}

fn write_set_schema<T: CacheSchema>(w: &mut SchemaWriter) {
    w.seq_begin("Set");
    T::write_schema(w);
    w.seq_end();
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
    use std::{
        borrow::Cow,
        collections::{BTreeMap, BTreeSet, HashMap, HashSet},
        num::{NonZeroU32, Saturating, Wrapping},
    };

    use super::*;
    use crate::schema_fingerprint;

    #[test]
    fn fingerprint_is_deterministic() {
        assert_eq!(
            schema_fingerprint::<(u32, bool)>(),
            schema_fingerprint::<(u32, bool)>()
        );
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
    fn borrowed_forms_are_schema_transparent() {
        assert_eq!(
            schema_fingerprint::<&'static u32>(),
            schema_fingerprint::<u32>()
        );
        assert_eq!(
            schema_fingerprint::<&'static mut u32>(),
            schema_fingerprint::<u32>()
        );
        assert_eq!(
            schema_fingerprint::<Cow<'static, u32>>(),
            schema_fingerprint::<u32>()
        );
    }

    #[test]
    fn borrowed_text_matches_owned_text_schema() {
        assert_eq!(schema_fingerprint::<str>(), schema_fingerprint::<String>());
        assert_eq!(
            schema_fingerprint::<&'static str>(),
            schema_fingerprint::<String>()
        );
        assert_eq!(
            schema_fingerprint::<Cow<'static, str>>(),
            schema_fingerprint::<String>()
        );
    }

    #[test]
    fn borrowed_slices_match_vec_schema() {
        assert_eq!(
            schema_fingerprint::<[u32]>(),
            schema_fingerprint::<Vec<u32>>()
        );
        assert_eq!(
            schema_fingerprint::<&'static [u32]>(),
            schema_fingerprint::<Vec<u32>>()
        );
        assert_eq!(
            schema_fingerprint::<Cow<'static, [u32]>>(),
            schema_fingerprint::<Vec<u32>>()
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
    fn unit_schema_is_distinct_from_single_empty_tuple_like_field() {
        assert_ne!(schema_fingerprint::<()>(), schema_fingerprint::<((),)>());
    }

    #[test]
    fn map_collections_use_logical_map_schema() {
        assert_eq!(
            schema_fingerprint::<HashMap<u32, u64>>(),
            schema_fingerprint::<BTreeMap<u32, u64>>()
        );
        assert_ne!(
            schema_fingerprint::<HashMap<u32, u64>>(),
            schema_fingerprint::<HashMap<u64, u64>>()
        );
        assert_ne!(
            schema_fingerprint::<HashMap<u32, u64>>(),
            schema_fingerprint::<HashMap<u32, u32>>()
        );
    }

    #[test]
    fn set_collections_use_logical_set_schema() {
        assert_eq!(
            schema_fingerprint::<HashSet<u32>>(),
            schema_fingerprint::<BTreeSet<u32>>()
        );
        assert_ne!(
            schema_fingerprint::<HashSet<u32>>(),
            schema_fingerprint::<HashSet<u64>>()
        );
        assert_ne!(
            schema_fingerprint::<HashSet<u32>>(),
            schema_fingerprint::<Vec<u32>>()
        );
    }

    #[test]
    fn arithmetic_wrappers_use_inner_numeric_schema() {
        assert_eq!(
            schema_fingerprint::<Wrapping<u32>>(),
            schema_fingerprint::<u32>()
        );
        assert_eq!(
            schema_fingerprint::<Saturating<u32>>(),
            schema_fingerprint::<u32>()
        );
    }

    #[test]
    fn nonzero_wrappers_use_distinct_schema() {
        assert_eq!(
            schema_fingerprint::<NonZeroU32>(),
            schema_fingerprint::<NonZeroU32>()
        );
        assert_ne!(
            schema_fingerprint::<NonZeroU32>(),
            schema_fingerprint::<u32>()
        );
    }
}
