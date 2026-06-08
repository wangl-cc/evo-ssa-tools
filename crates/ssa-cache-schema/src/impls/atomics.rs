use crate::{CacheSchema, SchemaWriter};

macro_rules! impl_atomic_schema {
    ($(#[$cfg:meta] $ty:ty => $name:literal),+ $(,)?) => {
        $(
            #[$cfg]
            impl CacheSchema for $ty {
                fn write_schema(w: &mut SchemaWriter) {
                    w.primitive($name);
                }
            }
        )+
    };
}

impl_atomic_schema!(
    #[cfg(target_has_atomic = "8")] std::sync::atomic::AtomicBool => "AtomicBool",
    #[cfg(target_has_atomic = "8")] std::sync::atomic::AtomicU8 => "AtomicU8",
    #[cfg(target_has_atomic = "16")] std::sync::atomic::AtomicU16 => "AtomicU16",
    #[cfg(target_has_atomic = "32")] std::sync::atomic::AtomicU32 => "AtomicU32",
    #[cfg(target_has_atomic = "64")] std::sync::atomic::AtomicU64 => "AtomicU64",
    #[cfg(target_has_atomic = "ptr")] std::sync::atomic::AtomicUsize => "AtomicUsize",
    #[cfg(target_has_atomic = "8")] std::sync::atomic::AtomicI8 => "AtomicI8",
    #[cfg(target_has_atomic = "16")] std::sync::atomic::AtomicI16 => "AtomicI16",
    #[cfg(target_has_atomic = "32")] std::sync::atomic::AtomicI32 => "AtomicI32",
    #[cfg(target_has_atomic = "64")] std::sync::atomic::AtomicI64 => "AtomicI64",
    #[cfg(target_has_atomic = "ptr")] std::sync::atomic::AtomicIsize => "AtomicIsize",
);

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use crate::schema_fingerprint;

    #[cfg(target_has_atomic = "32")]
    #[test]
    fn atomic_integer_schema_is_distinct_from_inner_integer() {
        assert_eq!(
            schema_fingerprint::<std::sync::atomic::AtomicU32>(),
            schema_fingerprint::<std::sync::atomic::AtomicU32>()
        );
        assert_ne!(
            schema_fingerprint::<std::sync::atomic::AtomicU32>(),
            schema_fingerprint::<u32>()
        );
    }

    #[cfg(target_has_atomic = "8")]
    #[test]
    fn atomic_bool_schema_is_distinct_from_bool() {
        assert_eq!(
            schema_fingerprint::<std::sync::atomic::AtomicBool>(),
            schema_fingerprint::<std::sync::atomic::AtomicBool>()
        );
        assert_ne!(
            schema_fingerprint::<std::sync::atomic::AtomicBool>(),
            schema_fingerprint::<bool>()
        );
    }
}
