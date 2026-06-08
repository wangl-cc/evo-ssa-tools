use crate::{CacheSchema, SchemaWriter};

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
        w.seq_begin("Sequence");
        T::write_schema(w);
        w.seq_end();
    }
}

impl<T: CacheSchema, const N: usize> CacheSchema for [T; N] {
    fn write_schema(w: &mut SchemaWriter) {
        w.array_begin(N);
        T::write_schema(w);
        w.array_end();
    }
}

macro_rules! impl_map_schema {
    (impl<$K:ident, $V:ident $(, $extra:ident)*> for $ty:ty) => {
        impl<$K: CacheSchema, $V: CacheSchema $(, $extra)*> CacheSchema for $ty {
            fn write_schema(w: &mut SchemaWriter) {
                w.map_begin("Map");
                $K::write_schema(w);
                $V::write_schema(w);
                w.map_end();
            }
        }
    };
}

macro_rules! impl_set_schema {
    (impl<$T:ident $(, $extra:ident)*> for $ty:ty) => {
        impl<$T: CacheSchema $(, $extra)*> CacheSchema for $ty {
            fn write_schema(w: &mut SchemaWriter) {
                w.seq_begin("Set");
                $T::write_schema(w);
                w.seq_end();
            }
        }
    };
}

impl_map_schema!(impl<K, V, S> for std::collections::HashMap<K, V, S>);
impl_map_schema!(impl<K, V> for std::collections::BTreeMap<K, V>);

impl_set_schema!(impl<T, S> for std::collections::HashSet<T, S>);
impl_set_schema!(impl<T> for std::collections::BTreeSet<T>);

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

    use crate::schema_fingerprint;

    #[test]
    fn string_and_str_share_text_schema() {
        assert_eq!(schema_fingerprint::<str>(), schema_fingerprint::<String>());
    }

    #[test]
    fn string_schema_is_distinct_from_raw_byte_sequence() {
        assert_ne!(
            schema_fingerprint::<String>(),
            schema_fingerprint::<Vec<u8>>()
        );
    }

    #[test]
    fn sequence_schema_includes_inner_type() {
        assert_ne!(
            schema_fingerprint::<Vec<u32>>(),
            schema_fingerprint::<Vec<u64>>()
        );
    }

    #[test]
    fn slice_and_vec_share_sequence_schema() {
        assert_eq!(
            schema_fingerprint::<[u32]>(),
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
}
