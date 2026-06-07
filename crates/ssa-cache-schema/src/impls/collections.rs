use crate::{CacheSchema, SchemaWriter};

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
