use ssa_cache_schema::CacheSchema;

#[derive(CacheSchema)]
struct Bad {
    #[cache_schema(rename = "first", rename = "second")]
    value: u32,
}

fn main() {}
