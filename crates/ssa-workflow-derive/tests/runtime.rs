use workflow::cache::{CacheSchema, CanonicalEncode};

#[derive(ssa_workflow_derive::CacheSchema, ssa_workflow_derive::CanonicalEncode)]
#[cache_schema(version = 1, crate = "workflow")]
#[canonical_encode(crate = "workflow")]
struct NamedKey<T> {
    generation: u64,
    selection: T,
    counts: [u16; 2],
}

#[derive(ssa_workflow_derive::CacheSchema, ssa_workflow_derive::CanonicalEncode)]
#[cache_schema(version = 1, crate = "workflow")]
#[canonical_encode(crate = "workflow")]
struct TupleKey(u16, u64);

#[derive(ssa_workflow_derive::CacheSchema, ssa_workflow_derive::CanonicalEncode)]
#[allow(dead_code)]
#[cache_schema(version = 1, crate = "workflow")]
#[canonical_encode(crate = "workflow")]
struct UnitKey;

#[derive(ssa_workflow_derive::CacheSchema)]
#[cache_schema(version = 1, crate = "workflow")]
struct OutputValue {
    total: f64,
    label: String,
}

#[derive(ssa_workflow_derive::CacheSchema, ssa_workflow_derive::CanonicalEncode)]
#[cache_schema(version = 1, crate = "workflow")]
#[canonical_encode(crate = "workflow")]
enum OptionKey {
    Fast,
    Accurate,
    Slow,
}

mod version_one {
    #[derive(ssa_workflow_derive::CacheSchema, ssa_workflow_derive::CanonicalEncode)]
    #[cache_schema(version = 1, crate = "workflow")]
    #[canonical_encode(crate = "workflow")]
    pub struct Key {
        pub value: u64,
    }
}

mod version_two {
    #[derive(ssa_workflow_derive::CacheSchema, ssa_workflow_derive::CanonicalEncode)]
    #[cache_schema(version = 2, crate = "workflow")]
    #[canonical_encode(crate = "workflow")]
    pub struct Key {
        pub value: u64,
    }
}

mod field_name_one {
    #[derive(ssa_workflow_derive::CacheSchema, ssa_workflow_derive::CanonicalEncode)]
    #[cache_schema(version = 1, crate = "workflow")]
    #[canonical_encode(crate = "workflow")]
    pub struct Key {
        pub birth_rate: f64,
        pub death_rate: f64,
    }
}

mod field_name_two {
    #[derive(ssa_workflow_derive::CacheSchema, ssa_workflow_derive::CanonicalEncode)]
    #[cache_schema(version = 1, crate = "workflow")]
    #[canonical_encode(crate = "workflow")]
    pub struct Key {
        pub lambda: f64,
        pub death_rate: f64,
    }
}

mod option_order_one {
    #[derive(ssa_workflow_derive::CacheSchema, ssa_workflow_derive::CanonicalEncode)]
    #[cache_schema(version = 1, crate = "workflow")]
    #[canonical_encode(crate = "workflow")]
    pub enum Key {
        Alpha,
        Beta,
    }
}

mod option_order_two {
    #[derive(ssa_workflow_derive::CacheSchema, ssa_workflow_derive::CanonicalEncode)]
    #[cache_schema(version = 1, crate = "workflow")]
    #[canonical_encode(crate = "workflow")]
    pub enum Key {
        Beta,
        Alpha,
    }
}

mod option_append_one {
    #[derive(ssa_workflow_derive::CacheSchema, ssa_workflow_derive::CanonicalEncode)]
    #[cache_schema(version = 1, crate = "workflow")]
    #[canonical_encode(crate = "workflow")]
    pub enum Key {
        Alpha,
        Beta,
    }
}

mod option_append_two {
    #[derive(ssa_workflow_derive::CacheSchema, ssa_workflow_derive::CanonicalEncode)]
    #[cache_schema(version = 1, crate = "workflow")]
    #[canonical_encode(crate = "workflow")]
    pub enum Key {
        Alpha,
        Beta,
        Gamma,
    }
}

#[test]
fn derive_named_struct_adds_trait_bounds_and_encodes_in_field_order() {
    type Key = NamedKey<f64>;

    let value = Key {
        generation: 0x0102_0304_0506_0708,
        selection: -0.0,
        counts: [0x090a, 0x0b0c],
    };
    let mut buffer = vec![0u8; Key::SIZE];
    let encoded = unsafe { value.encode_with_buffer(&mut buffer) };

    let expected = [
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x09, 0x0a, 0x0b, 0x0c,
    ];

    assert_eq!(Key::SIZE, expected.len());
    assert_eq!(encoded, expected);
}

#[test]
fn derive_tuple_struct_uses_positional_fields() {
    let value = TupleKey(0x1112, 0x1314_1516_1718_191a);
    let mut buffer = vec![0u8; TupleKey::SIZE];
    let encoded = unsafe { value.encode_with_buffer(&mut buffer) };

    let expected = [0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a];

    assert_eq!(TupleKey::SIZE, expected.len());
    assert_eq!(encoded, expected);
}

#[test]
fn derive_unit_struct_has_zero_size() {
    let value = UnitKey;
    let mut buffer = [];
    let encoded = unsafe { value.encode_with_buffer(&mut buffer) };

    assert_eq!(UnitKey::SIZE, 0);
    assert!(encoded.is_empty());
}

#[test]
fn derive_cache_schema_supports_output_only_types() {
    let value = OutputValue {
        total: 1.25,
        label: "cached".to_owned(),
    };

    assert_eq!(value.total, 1.25);
    assert_eq!(value.label, "cached");
    assert_ne!(OutputValue::SCHEMA_SIGNATURE, 0);
}

#[test]
fn derive_unit_enum_encodes_declaration_order_as_u8() {
    let mut buffer = [0xff; OptionKey::SIZE];

    assert_eq!(OptionKey::SIZE, 1);
    assert_eq!(
        unsafe { OptionKey::Fast.encode_with_buffer(&mut buffer) },
        [0]
    );
    assert_eq!(
        unsafe { OptionKey::Accurate.encode_with_buffer(&mut buffer) },
        [1]
    );
    assert_eq!(
        unsafe { OptionKey::Slow.encode_with_buffer(&mut buffer) },
        [2]
    );
}

#[test]
fn derive_version_changes_schema_signature() {
    assert_ne!(
        version_one::Key::SCHEMA_SIGNATURE,
        version_two::Key::SCHEMA_SIGNATURE,
        "schema version changes must select a different persistent cache namespace"
    );
}

#[test]
fn derive_field_name_changes_schema_signature() {
    let _ = field_name_one::Key {
        birth_rate: 1.0,
        death_rate: 2.0,
    };
    let _ = field_name_two::Key {
        lambda: 1.0,
        death_rate: 2.0,
    };

    assert_ne!(
        field_name_one::Key::SCHEMA_SIGNATURE,
        field_name_two::Key::SCHEMA_SIGNATURE,
        "field names are part of the conservative derived schema contract"
    );
}

#[test]
fn derive_unit_enum_order_changes_schema_signature() {
    let _ = option_order_one::Key::Alpha;
    let _ = option_order_one::Key::Beta;
    let _ = option_order_two::Key::Beta;
    let _ = option_order_two::Key::Alpha;

    assert_ne!(
        option_order_one::Key::SCHEMA_SIGNATURE,
        option_order_two::Key::SCHEMA_SIGNATURE,
        "unit enum declaration order changes must select a different persistent cache namespace"
    );
}

#[test]
fn derive_unit_enum_append_changes_schema_signature() {
    let _ = option_append_one::Key::Alpha;
    let _ = option_append_one::Key::Beta;
    let _ = option_append_two::Key::Alpha;
    let _ = option_append_two::Key::Beta;
    let _ = option_append_two::Key::Gamma;

    assert_ne!(
        option_append_one::Key::SCHEMA_SIGNATURE,
        option_append_two::Key::SCHEMA_SIGNATURE,
        "unit enum variants are part of the conservative derived schema contract"
    );
}
