use super::{CacheSchema, schema_fingerprint};

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
