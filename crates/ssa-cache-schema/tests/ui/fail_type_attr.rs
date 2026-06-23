use ssa_cache_schema::CacheSchema;

#[derive(CacheSchema)]
#[cache_schema(skip)]
struct Bad {
    value: u32,
}

fn main() {}
