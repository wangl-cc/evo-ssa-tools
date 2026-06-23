use ssa_cache_schema::CacheSchema;

#[derive(CacheSchema)]
#[cache_schema(rename = "First", rename = "Second")]
struct Bad {
    value: u32,
}

fn main() {}
