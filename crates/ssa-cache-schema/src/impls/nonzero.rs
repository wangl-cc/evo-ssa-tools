use crate::{CacheSchema, SchemaWriter};

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
