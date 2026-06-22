use super::{CacheSchema, schema_fingerprint};

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
