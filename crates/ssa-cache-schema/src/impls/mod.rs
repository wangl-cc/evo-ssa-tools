mod containers;
mod scalars;
mod tuples;

use std::marker::PhantomData;

use crate::{CacheSchema, SchemaWriter};

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

impl<T: ?Sized> CacheSchema for PhantomData<T> {
    fn write_schema(w: &mut SchemaWriter) {
        w.primitive("PhantomData");
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use std::{borrow::Cow, marker::PhantomData};

    use crate::schema_fingerprint;

    #[test]
    fn option_schema_includes_inner_type() {
        assert_ne!(
            schema_fingerprint::<Option<u32>>(),
            schema_fingerprint::<Option<u64>>()
        );
    }

    #[test]
    fn result_schema_includes_ok_and_err_types() {
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
    fn ownership_wrappers_are_schema_transparent() {
        assert_eq!(
            schema_fingerprint::<Box<u32>>(),
            schema_fingerprint::<u32>()
        );
        assert_eq!(
            schema_fingerprint::<Box<Box<u32>>>(),
            schema_fingerprint::<u32>()
        );
        assert_ne!(
            schema_fingerprint::<Box<u32>>(),
            schema_fingerprint::<Box<u64>>()
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
            schema_fingerprint::<&'static [u32]>(),
            schema_fingerprint::<Vec<u32>>()
        );
        assert_eq!(
            schema_fingerprint::<Cow<'static, [u32]>>(),
            schema_fingerprint::<Vec<u32>>()
        );
        assert_eq!(
            schema_fingerprint::<Box<[u32]>>(),
            schema_fingerprint::<Vec<u32>>()
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
    fn phantom_data_accepts_unsized_type_parameter() {
        assert_eq!(
            schema_fingerprint::<PhantomData<str>>(),
            schema_fingerprint::<PhantomData<u32>>()
        );
    }
}
