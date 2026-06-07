use crate::{CacheSchema, SchemaWriter};

impl CacheSchema for String {
    fn write_schema(w: &mut SchemaWriter) {
        str::write_schema(w);
    }
}

impl CacheSchema for str {
    fn write_schema(w: &mut SchemaWriter) {
        w.seq_begin("String");
        u8::write_schema(w);
        w.seq_end();
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use crate::schema_fingerprint;

    #[test]
    fn string_and_str_share_text_schema() {
        assert_eq!(schema_fingerprint::<str>(), schema_fingerprint::<String>());
    }

    #[test]
    fn string_schema_is_distinct_from_raw_byte_sequence() {
        assert_ne!(
            schema_fingerprint::<String>(),
            schema_fingerprint::<Vec<u8>>()
        );
    }
}
