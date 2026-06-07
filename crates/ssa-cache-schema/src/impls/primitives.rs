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

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use std::num::NonZeroU32;

    use crate::schema_fingerprint;

    #[test]
    fn type_alias_uses_aliased_type_schema() {
        type MyU64 = u64;

        assert_eq!(schema_fingerprint::<MyU64>(), schema_fingerprint::<u64>());
    }

    #[test]
    fn unit_schema_is_distinct_from_single_empty_tuple_like_field() {
        assert_ne!(schema_fingerprint::<()>(), schema_fingerprint::<((),)>());
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
