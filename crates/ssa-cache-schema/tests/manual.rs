use ssa_cache_schema::{CacheSchema, EmptyProductStyle, SchemaWriter, schema_fingerprint};

#[expect(
    dead_code,
    reason = "manual schema fixture is fingerprinted by type and never constructed"
)]
struct ManualRecord {
    value: u32,
}

impl CacheSchema for ManualRecord {
    fn write_schema(w: &mut SchemaWriter) {
        w.struct_begin("ManualRecord");
        w.field_begin(0, Some("value"));
        u32::write_schema(w);
        w.field_end();
        w.struct_end();
    }
}

struct ManualUnit;

impl CacheSchema for ManualUnit {
    fn write_schema(w: &mut SchemaWriter) {
        w.struct_begin("ManualUnit");
        w.empty_product_style(EmptyProductStyle::Unit);
        w.struct_end();
    }
}

#[test]
fn manual_implementation_uses_public_writer_api() {
    assert_ne!(
        schema_fingerprint::<ManualRecord>(),
        schema_fingerprint::<ManualUnit>()
    );
}
