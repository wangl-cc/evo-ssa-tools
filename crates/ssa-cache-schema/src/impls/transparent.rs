use std::marker::PhantomData;

use crate::{CacheSchema, SchemaWriter};

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

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use std::{
        borrow::Cow,
        num::{Saturating, Wrapping},
    };

    use super::*;
    use crate::schema_fingerprint;

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
}
