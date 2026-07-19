use super::{CacheSchema, schema_fingerprint};

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
