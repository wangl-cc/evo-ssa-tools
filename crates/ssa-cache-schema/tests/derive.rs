#![allow(
    dead_code,
    reason = "derive fixtures are fingerprinted by type and intentionally never constructed"
)]

use std::marker::PhantomData;

use ssa_cache_schema::{CacheSchema, schema_fingerprint};

#[test]
fn default_type_name_affects_fingerprint() {
    #[derive(CacheSchema)]
    struct First {
        value: u32,
    }

    #[derive(CacheSchema)]
    struct Second {
        value: u32,
    }

    mod moved {
        use ssa_cache_schema::CacheSchema;

        #[derive(CacheSchema)]
        pub(super) struct First {
            pub(super) value: u32,
        }
    }

    assert_ne!(
        schema_fingerprint::<First>(),
        schema_fingerprint::<Second>()
    );
}

#[test]
fn module_move_does_not_affect_fingerprint() {
    #[derive(CacheSchema)]
    struct Params {
        value: u32,
    }

    mod moved {
        use ssa_cache_schema::CacheSchema;

        #[derive(CacheSchema)]
        pub(super) struct Params {
            pub(super) value: u32,
        }
    }

    assert_eq!(
        schema_fingerprint::<Params>(),
        schema_fingerprint::<moved::Params>()
    );
}

#[test]
fn type_rust_rename_can_keep_schema_name() {
    #[derive(CacheSchema)]
    #[cache_schema(rename = "Original")]
    struct Original {
        value: u32,
    }

    #[derive(CacheSchema)]
    #[cache_schema(rename = "Original")]
    struct Renamed {
        value: u32,
    }

    assert_eq!(
        schema_fingerprint::<Original>(),
        schema_fingerprint::<Renamed>()
    );
}

#[test]
#[allow(non_camel_case_types, reason = "raw identifier fixture uses a keyword")]
fn raw_identifiers_use_unescaped_default_schema_names() {
    #[derive(CacheSchema)]
    struct r#type {
        r#gen: u32,
    }

    #[derive(CacheSchema)]
    #[cache_schema(rename = "type")]
    struct ExplicitTypeAndField {
        #[cache_schema(rename = "gen")]
        value: u32,
    }

    #[derive(CacheSchema)]
    #[cache_schema(rename = "RawVariant")]
    enum RawVariant {
        r#gen,
    }

    #[derive(CacheSchema)]
    #[cache_schema(rename = "RawVariant")]
    enum ExplicitVariant {
        #[cache_schema(rename = "gen")]
        Other,
    }

    assert_eq!(
        schema_fingerprint::<r#type>(),
        schema_fingerprint::<ExplicitTypeAndField>()
    );
    assert_eq!(
        schema_fingerprint::<RawVariant>(),
        schema_fingerprint::<ExplicitVariant>()
    );
}

#[test]
fn field_rust_rename_without_schema_rename_changes_fingerprint() {
    #[derive(CacheSchema)]
    #[cache_schema(rename = "Shape")]
    struct Old {
        width: u32,
    }

    #[derive(CacheSchema)]
    #[cache_schema(rename = "Shape")]
    struct New {
        w: u32,
    }

    assert_ne!(schema_fingerprint::<Old>(), schema_fingerprint::<New>());
}

#[test]
fn field_rust_rename_can_keep_schema_name() {
    #[derive(CacheSchema)]
    #[cache_schema(rename = "Shape")]
    struct Old {
        width: u32,
    }

    #[derive(CacheSchema)]
    #[cache_schema(rename = "Shape")]
    struct New {
        #[cache_schema(rename = "width")]
        w: u32,
    }

    assert_eq!(schema_fingerprint::<Old>(), schema_fingerprint::<New>());
}

#[test]
fn field_reorder_changes_fingerprint() {
    #[derive(CacheSchema)]
    #[cache_schema(rename = "Shape")]
    struct WidthHeight {
        width: u32,
        height: u32,
    }

    #[derive(CacheSchema)]
    #[cache_schema(rename = "Shape")]
    struct HeightWidth {
        height: u32,
        width: u32,
    }

    assert_ne!(
        schema_fingerprint::<WidthHeight>(),
        schema_fingerprint::<HeightWidth>()
    );
}

#[test]
fn field_add_remove_changes_fingerprint() {
    #[derive(CacheSchema)]
    #[cache_schema(rename = "Shape")]
    struct OneField {
        width: u32,
    }

    #[derive(CacheSchema)]
    #[cache_schema(rename = "Shape")]
    struct TwoFields {
        width: u32,
        height: u32,
    }

    assert_ne!(
        schema_fingerprint::<OneField>(),
        schema_fingerprint::<TwoFields>()
    );
}

#[test]
fn field_type_change_changes_fingerprint() {
    #[derive(CacheSchema)]
    #[cache_schema(rename = "Shape")]
    struct U32Width {
        width: u32,
    }

    #[derive(CacheSchema)]
    #[cache_schema(rename = "Shape")]
    struct U64Width {
        width: u64,
    }

    assert_ne!(
        schema_fingerprint::<U32Width>(),
        schema_fingerprint::<U64Width>()
    );
}

#[test]
fn named_and_tuple_fields_are_distinct() {
    #[derive(CacheSchema)]
    #[cache_schema(rename = "Shape")]
    struct Named {
        value: u32,
    }

    #[derive(CacheSchema)]
    #[cache_schema(rename = "Shape")]
    struct Tuple(u32);

    assert_ne!(schema_fingerprint::<Named>(), schema_fingerprint::<Tuple>());
}

#[test]
fn empty_product_shapes_are_distinct() {
    #[derive(CacheSchema)]
    #[cache_schema(rename = "Empty")]
    struct Unit;

    #[derive(CacheSchema)]
    #[cache_schema(rename = "Empty")]
    struct Tuple();

    #[derive(CacheSchema)]
    #[cache_schema(rename = "Empty")]
    struct Named {}

    assert_ne!(schema_fingerprint::<Unit>(), schema_fingerprint::<Tuple>());
    assert_ne!(schema_fingerprint::<Unit>(), schema_fingerprint::<Named>());
    assert_ne!(schema_fingerprint::<Tuple>(), schema_fingerprint::<Named>());

    #[derive(CacheSchema)]
    #[cache_schema(rename = "Event")]
    enum UnitVariant {
        Empty,
    }

    #[derive(CacheSchema)]
    #[cache_schema(rename = "Event")]
    enum TupleVariant {
        Empty(),
    }

    #[derive(CacheSchema)]
    #[cache_schema(rename = "Event")]
    enum NamedVariant {
        Empty {},
    }

    assert_ne!(
        schema_fingerprint::<UnitVariant>(),
        schema_fingerprint::<TupleVariant>()
    );
    assert_ne!(
        schema_fingerprint::<UnitVariant>(),
        schema_fingerprint::<NamedVariant>()
    );
    assert_ne!(
        schema_fingerprint::<TupleVariant>(),
        schema_fingerprint::<NamedVariant>()
    );
}

#[test]
fn standard_option_matches_derive_equivalent_schema() {
    #[derive(CacheSchema)]
    #[cache_schema(rename = "Option")]
    enum DerivedOption<T> {
        None,
        Some(T),
    }

    assert_eq!(
        schema_fingerprint::<Option<u32>>(),
        schema_fingerprint::<DerivedOption<u32>>()
    );
}

#[test]
fn tuple_field_rename_can_name_wire_field() {
    #[derive(CacheSchema)]
    #[cache_schema(rename = "Shape")]
    struct Named {
        value: u32,
    }

    #[derive(CacheSchema)]
    #[cache_schema(rename = "Shape")]
    struct Tuple(#[cache_schema(rename = "value")] u32);

    assert_eq!(schema_fingerprint::<Named>(), schema_fingerprint::<Tuple>());
}

#[test]
fn variant_rust_rename_without_schema_rename_changes_fingerprint() {
    #[derive(CacheSchema)]
    #[cache_schema(rename = "Event")]
    enum Old {
        Created,
    }

    #[derive(CacheSchema)]
    #[cache_schema(rename = "Event")]
    enum New {
        Made,
    }

    assert_ne!(schema_fingerprint::<Old>(), schema_fingerprint::<New>());
}

#[test]
fn variant_rust_rename_can_keep_schema_name() {
    #[derive(CacheSchema)]
    #[cache_schema(rename = "Event")]
    enum Old {
        Created { id: u64 },
    }

    #[derive(CacheSchema)]
    #[cache_schema(rename = "Event")]
    enum New {
        #[cache_schema(rename = "Created")]
        Made { id: u64 },
    }

    assert_eq!(schema_fingerprint::<Old>(), schema_fingerprint::<New>());
}

#[test]
fn variant_reorder_changes_fingerprint() {
    #[derive(CacheSchema)]
    #[cache_schema(rename = "Event")]
    enum First {
        Created,
        Deleted,
    }

    #[derive(CacheSchema)]
    #[cache_schema(rename = "Event")]
    enum Second {
        Deleted,
        Created,
    }

    assert_ne!(
        schema_fingerprint::<First>(),
        schema_fingerprint::<Second>()
    );
}

#[test]
fn explicit_enum_discriminants_do_not_affect_derived_schema() {
    #[derive(CacheSchema)]
    #[cache_schema(rename = "Event")]
    enum First {
        Created = 1,
        Deleted = 2,
    }

    #[derive(CacheSchema)]
    #[cache_schema(rename = "Event")]
    enum Second {
        Created = 10,
        Deleted = 20,
    }

    assert_eq!(
        schema_fingerprint::<First>(),
        schema_fingerprint::<Second>()
    );
}

#[test]
fn variant_add_remove_changes_fingerprint() {
    #[derive(CacheSchema)]
    #[cache_schema(rename = "Event")]
    enum One {
        Created,
    }

    #[derive(CacheSchema)]
    #[cache_schema(rename = "Event")]
    enum Two {
        Created,
        Deleted,
    }

    assert_ne!(schema_fingerprint::<One>(), schema_fingerprint::<Two>());
}

#[test]
fn variant_field_shape_changes_fingerprint() {
    #[derive(CacheSchema)]
    #[cache_schema(rename = "Event")]
    enum StructVariant {
        Created { id: u64 },
    }

    #[derive(CacheSchema)]
    #[cache_schema(rename = "Event")]
    enum TupleVariant {
        Created(u64),
    }

    #[derive(CacheSchema)]
    #[cache_schema(rename = "Event")]
    enum RenamedTupleVariant {
        Created(#[cache_schema(rename = "id")] u64),
    }

    #[derive(CacheSchema)]
    #[cache_schema(rename = "Event")]
    enum TypeChanged {
        Created { id: u32 },
    }

    assert_ne!(
        schema_fingerprint::<StructVariant>(),
        schema_fingerprint::<TupleVariant>()
    );
    assert_eq!(
        schema_fingerprint::<StructVariant>(),
        schema_fingerprint::<RenamedTupleVariant>()
    );
    assert_ne!(
        schema_fingerprint::<StructVariant>(),
        schema_fingerprint::<TypeChanged>()
    );
}

#[test]
fn nested_type_change_changes_outer_fingerprint() {
    #[derive(CacheSchema)]
    #[cache_schema(rename = "Inner")]
    struct InnerU32 {
        value: u32,
    }

    #[derive(CacheSchema)]
    #[cache_schema(rename = "Inner")]
    struct InnerU64 {
        value: u64,
    }

    #[derive(CacheSchema)]
    #[cache_schema(rename = "Outer")]
    struct OuterU32 {
        inner: InnerU32,
    }

    #[derive(CacheSchema)]
    #[cache_schema(rename = "Outer")]
    struct OuterU64 {
        inner: InnerU64,
    }

    assert_ne!(
        schema_fingerprint::<OuterU32>(),
        schema_fingerprint::<OuterU64>()
    );
}

#[test]
fn type_version_changes_fingerprint() {
    #[derive(CacheSchema)]
    #[cache_schema(rename = "Versioned", version = "v1")]
    struct V1 {
        value: u32,
    }

    #[derive(CacheSchema)]
    #[cache_schema(rename = "Versioned", version = "v2")]
    struct V2 {
        value: u32,
    }

    assert_ne!(schema_fingerprint::<V1>(), schema_fingerprint::<V2>());
}

#[test]
fn schema_fingerprints_match_golden_vectors() {
    #[derive(CacheSchema)]
    #[cache_schema(rename = "GoldenStruct")]
    struct GoldenStruct {
        width: u32,
        heights: Vec<u16>,
    }

    #[derive(CacheSchema)]
    #[cache_schema(rename = "GoldenEnum")]
    enum GoldenEnum {
        Empty,
        Value { id: u64 },
    }

    #[derive(CacheSchema)]
    #[cache_schema(rename = "GoldenVersioned", version = "v1")]
    struct GoldenVersioned {
        value: u8,
    }

    #[derive(CacheSchema)]
    #[cache_schema(rename = "GoldenEmpty")]
    struct GoldenUnit;

    assert_eq!(schema_fingerprint::<u32>(), [
        176, 250, 45, 168, 40, 240, 228, 73, 44, 176, 60, 161, 26, 120, 162, 78
    ]);
    assert_eq!(schema_fingerprint::<Vec<u32>>(), [
        22, 240, 44, 117, 234, 56, 150, 21, 192, 237, 101, 197, 132, 69, 134, 41
    ]);
    assert_eq!(schema_fingerprint::<GoldenStruct>(), [
        122, 179, 166, 139, 48, 98, 88, 125, 104, 38, 59, 116, 5, 10, 227, 57
    ]);
    assert_eq!(schema_fingerprint::<GoldenEnum>(), [
        113, 194, 83, 155, 51, 236, 0, 224, 175, 218, 48, 145, 183, 219, 23, 33
    ]);
    assert_eq!(schema_fingerprint::<GoldenVersioned>(), [
        211, 152, 5, 168, 201, 103, 34, 197, 12, 214, 218, 61, 66, 107, 107, 80
    ]);
    assert_eq!(schema_fingerprint::<GoldenUnit>(), [
        1, 103, 239, 86, 59, 158, 61, 140, 198, 139, 114, 163, 78, 184, 107, 114
    ]);
}

#[test]
fn serde_attrs_do_not_affect_fingerprint() {
    #[derive(CacheSchema)]
    #[cache_schema(rename = "SerdeIgnored")]
    struct WithSerdeAttrs {
        #[serde(rename = "wire_width", skip)]
        width: u32,
    }

    #[derive(CacheSchema)]
    #[cache_schema(rename = "SerdeIgnored")]
    struct Plain {
        width: u32,
    }

    #[derive(CacheSchema)]
    #[cache_schema(rename = "SerdeIgnoredEvent")]
    enum WithSerdeVariantAttrs {
        #[serde(rename = "wire_created")]
        Created {
            #[serde(rename = "wire_id", skip)]
            id: u64,
        },
    }

    #[derive(CacheSchema)]
    #[cache_schema(rename = "SerdeIgnoredEvent")]
    enum PlainVariant {
        Created { id: u64 },
    }

    assert_eq!(
        schema_fingerprint::<WithSerdeAttrs>(),
        schema_fingerprint::<Plain>()
    );
    assert_eq!(
        schema_fingerprint::<WithSerdeVariantAttrs>(),
        schema_fingerprint::<PlainVariant>()
    );
}

#[test]
fn non_cache_schema_attrs_do_not_affect_fingerprint() {
    #[doc = "ignored by CacheSchema"]
    #[derive(CacheSchema)]
    #[cache_schema(rename = "Documented")]
    struct WithAttrs {
        #[doc = "also ignored"]
        value: u32,
    }

    #[derive(CacheSchema)]
    #[cache_schema(rename = "Documented")]
    struct Plain {
        value: u32,
    }

    #[derive(CacheSchema)]
    #[cache_schema(rename = "DocumentedEvent")]
    enum WithVariantAttrs {
        #[doc = "ignored variant documentation"]
        Created {
            #[doc = "ignored field documentation"]
            id: u64,
        },
    }

    #[derive(CacheSchema)]
    #[cache_schema(rename = "DocumentedEvent")]
    enum PlainVariant {
        Created { id: u64 },
    }

    assert_eq!(
        schema_fingerprint::<WithAttrs>(),
        schema_fingerprint::<Plain>()
    );
    assert_eq!(
        schema_fingerprint::<WithVariantAttrs>(),
        schema_fingerprint::<PlainVariant>()
    );
}

#[test]
fn supports_tuple_unit_generic_and_phantom_shapes() {
    #[derive(CacheSchema)]
    #[cache_schema(rename = "Tuple")]
    struct Tuple<T>(T, String);

    #[derive(CacheSchema)]
    #[cache_schema(rename = "Unit")]
    struct Unit;

    #[derive(CacheSchema)]
    #[cache_schema(rename = "WithPhantom")]
    struct WithPhantom<T> {
        value: u32,
        marker: PhantomData<T>,
    }

    let tuple = schema_fingerprint::<Tuple<u32>>();
    let unit = schema_fingerprint::<Unit>();
    let phantom_u32 = schema_fingerprint::<WithPhantom<u32>>();
    let phantom_u64 = schema_fingerprint::<WithPhantom<u64>>();

    assert_ne!(tuple, unit);
    assert_eq!(phantom_u32, phantom_u64);
}

#[test]
fn supports_lifetime_const_generics_and_where_clauses() {
    #[derive(CacheSchema)]
    #[cache_schema(rename = "BorrowedArray")]
    struct BorrowedArray<'a, T, const N: usize>
    where
        T: 'a,
    {
        values: [T; N],
        marker: PhantomData<&'a T>,
    }

    assert_ne!(
        schema_fingerprint::<BorrowedArray<'static, u32, 2>>(),
        schema_fingerprint::<BorrowedArray<'static, u32, 3>>()
    );
    assert_ne!(
        schema_fingerprint::<BorrowedArray<'static, u32, 2>>(),
        schema_fingerprint::<BorrowedArray<'static, u64, 2>>()
    );
}
