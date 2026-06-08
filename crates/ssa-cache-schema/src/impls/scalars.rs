use crate::{CacheSchema, SchemaWriter};

macro_rules! impl_scalar_schema {
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

impl_scalar_schema!(
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

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use crate::schema_fingerprint;

    #[test]
    fn type_alias_uses_aliased_type_schema() {
        type MyU64 = u64;

        assert_eq!(schema_fingerprint::<MyU64>(), schema_fingerprint::<u64>());
    }
}
