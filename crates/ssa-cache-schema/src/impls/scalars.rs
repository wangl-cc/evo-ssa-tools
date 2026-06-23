use crate::{CacheSchema, SchemaWriter};

macro_rules! impl_scalar_schema {
    ($($name:ident),+ $(,)?) => {
        $(
            impl CacheSchema for $name {
                fn write_schema(w: &mut SchemaWriter) {
                    w.leaf(stringify!($name));
                }
            }
        )+
    };
}

impl_scalar_schema!(
    bool, char, u8, u16, u32, u64, u128, i8, i16, i32, i64, i128, f32, f64,
);

macro_rules! impl_pointer_width_scalar_schema {
    ($($name:ident),+ $(,)?) => {
        $(
            impl CacheSchema for $name {
                fn write_schema(w: &mut SchemaWriter) {
                    write_pointer_width_schema(w, stringify!($name));
                }
            }
        )+
    };
}

impl_pointer_width_scalar_schema!(usize, isize);

macro_rules! impl_nonzero_schema {
    ($($name:ident),+ $(,)?) => {
        $(
            impl CacheSchema for std::num::$name {
                fn write_schema(w: &mut SchemaWriter) {
                    w.leaf(stringify!($name));
                }
            }
        )+
    };
}

impl_nonzero_schema!(
    NonZeroU8,
    NonZeroU16,
    NonZeroU32,
    NonZeroU64,
    NonZeroU128,
    NonZeroI8,
    NonZeroI16,
    NonZeroI32,
    NonZeroI64,
    NonZeroI128,
);

macro_rules! impl_pointer_width_nonzero_schema {
    ($($name:ident),+ $(,)?) => {
        $(
            impl CacheSchema for std::num::$name {
                fn write_schema(w: &mut SchemaWriter) {
                    write_pointer_width_schema(w, stringify!($name));
                }
            }
        )+
    };
}

impl_pointer_width_nonzero_schema!(NonZeroUsize, NonZeroIsize);

macro_rules! impl_atomic_schema {
    ($($width:literal: $($name:ident),+;)+) => {
        $(
            $(
                #[cfg(target_has_atomic = $width)]
                impl CacheSchema for std::sync::atomic::$name {
                    fn write_schema(w: &mut SchemaWriter) {
                        w.leaf(stringify!($name));
                    }
                }
            )+
        )+
    };
}

impl_atomic_schema!(
    "8": AtomicBool, AtomicU8, AtomicI8;
    "16": AtomicU16, AtomicI16;
    "32": AtomicU32, AtomicI32;
    "64": AtomicU64, AtomicI64;
);

macro_rules! impl_pointer_width_atomic_schema {
    ($($name:ident),+ $(,)?) => {
        $(
            #[cfg(target_has_atomic = "ptr")]
            impl CacheSchema for std::sync::atomic::$name {
                fn write_schema(w: &mut SchemaWriter) {
                    write_pointer_width_schema(w, stringify!($name));
                }
            }
        )+
    };
}

impl_pointer_width_atomic_schema!(AtomicUsize, AtomicIsize);

fn write_pointer_width_schema(w: &mut SchemaWriter, name: &str) {
    w.leaf(name);
    w.type_version(POINTER_WIDTH_SCHEMA);
}

#[cfg(target_pointer_width = "16")]
const POINTER_WIDTH_SCHEMA: &str = "target_pointer_width=16";
#[cfg(target_pointer_width = "32")]
const POINTER_WIDTH_SCHEMA: &str = "target_pointer_width=32";
#[cfg(target_pointer_width = "64")]
const POINTER_WIDTH_SCHEMA: &str = "target_pointer_width=64";

#[cfg(not(any(
    target_pointer_width = "16",
    target_pointer_width = "32",
    target_pointer_width = "64"
)))]
compile_error!("unsupported target pointer width for ssa-cache-schema pointer-sized schemas");

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
    use std::num::{NonZeroU32, Saturating, Wrapping};

    use crate::{CacheSchema, SchemaWriter, schema_fingerprint};

    #[test]
    fn type_alias_uses_aliased_type_schema() {
        type MyU64 = u64;

        assert_eq!(schema_fingerprint::<MyU64>(), schema_fingerprint::<u64>());
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

    #[test]
    fn pointer_width_integer_schemas_include_target_width() {
        struct BareUsize;

        impl CacheSchema for BareUsize {
            fn write_schema(w: &mut SchemaWriter) {
                w.leaf("usize");
            }
        }

        struct ExpectedUsize;

        impl CacheSchema for ExpectedUsize {
            fn write_schema(w: &mut SchemaWriter) {
                w.leaf("usize");
                w.type_version(super::POINTER_WIDTH_SCHEMA);
            }
        }

        assert_eq!(
            schema_fingerprint::<usize>(),
            schema_fingerprint::<ExpectedUsize>()
        );
        assert_ne!(
            schema_fingerprint::<usize>(),
            schema_fingerprint::<BareUsize>()
        );
    }

    #[test]
    fn pointer_width_nonzero_schemas_include_target_width() {
        struct BareNonZeroUsize;

        impl CacheSchema for BareNonZeroUsize {
            fn write_schema(w: &mut SchemaWriter) {
                w.leaf("NonZeroUsize");
            }
        }

        assert_ne!(
            schema_fingerprint::<std::num::NonZeroUsize>(),
            schema_fingerprint::<BareNonZeroUsize>()
        );
    }

    #[cfg(target_has_atomic = "ptr")]
    #[test]
    fn pointer_width_atomic_schemas_include_target_width() {
        struct BareAtomicUsize;

        impl CacheSchema for BareAtomicUsize {
            fn write_schema(w: &mut SchemaWriter) {
                w.leaf("AtomicUsize");
            }
        }

        assert_ne!(
            schema_fingerprint::<std::sync::atomic::AtomicUsize>(),
            schema_fingerprint::<BareAtomicUsize>()
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
