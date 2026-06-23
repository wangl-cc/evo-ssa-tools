use crate::{CacheSchema, SchemaWriter};

impl CacheSchema for () {
    fn write_schema(w: &mut SchemaWriter) {
        w.tuple_begin();
        w.tuple_end();
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
    fn unit_schema_is_distinct_from_single_empty_tuple_like_field() {
        assert_ne!(schema_fingerprint::<()>(), schema_fingerprint::<((),)>());
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
