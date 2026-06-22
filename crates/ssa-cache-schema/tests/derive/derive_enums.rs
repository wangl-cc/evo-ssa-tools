use super::{CacheSchema, schema_fingerprint};

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
