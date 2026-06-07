use crate::{CacheSchema, SchemaWriter};

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
    use crate::schema_fingerprint;

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
}
