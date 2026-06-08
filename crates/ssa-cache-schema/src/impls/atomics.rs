use crate::{CacheSchema, SchemaWriter};

macro_rules! impl_atomic_schema {
    ($($width:tt: $($ty:ident),+;)+) => {
        $(
            impl_atomic_schema!(@group $width; $($ty),+);
        )+
    };
    (@group 8; $($ty:ident),+) => {
        $(impl_atomic_schema!(@impl "8"; $ty);)+
    };
    (@group 16; $($ty:ident),+) => {
        $(impl_atomic_schema!(@impl "16"; $ty);)+
    };
    (@group 32; $($ty:ident),+) => {
        $(impl_atomic_schema!(@impl "32"; $ty);)+
    };
    (@group 64; $($ty:ident),+) => {
        $(impl_atomic_schema!(@impl "64"; $ty);)+
    };
    (@group ptr; $($ty:ident),+) => {
        $(impl_atomic_schema!(@impl "ptr"; $ty);)+
    };
    (@impl $width:literal; $ty:ident) => {
        #[cfg(target_has_atomic = $width)]
        impl CacheSchema for std::sync::atomic::$ty {
            fn write_schema(w: &mut SchemaWriter) {
                w.primitive(stringify!($ty));
            }
        }
    };
}

impl_atomic_schema!(
    8: AtomicBool, AtomicU8, AtomicI8;
    16: AtomicU16, AtomicI16;
    32: AtomicU32, AtomicI32;
    64: AtomicU64, AtomicI64;
    ptr: AtomicUsize, AtomicIsize;
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
