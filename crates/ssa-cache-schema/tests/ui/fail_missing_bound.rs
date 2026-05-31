use ssa_cache_schema::{CacheSchema, schema_fingerprint};

struct NoSchema;

#[derive(CacheSchema)]
struct Wrapper<T> {
    value: T,
}

fn main() {
    let _ = schema_fingerprint::<Wrapper<NoSchema>>();
}
