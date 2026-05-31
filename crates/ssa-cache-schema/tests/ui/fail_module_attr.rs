use ssa_cache_schema::CacheSchema;

#[derive(CacheSchema)]
#[cache_schema(module = "old::module")]
struct Bad {
    value: u32,
}

fn main() {}
