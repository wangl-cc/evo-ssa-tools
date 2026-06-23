use ssa_cache_schema::CacheSchema;

#[derive(CacheSchema)]
enum Bad {
    #[cache_schema(rename = "First", rename = "Second")]
    Variant,
}

fn main() {}
