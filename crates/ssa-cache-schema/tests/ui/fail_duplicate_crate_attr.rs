use ssa_cache_schema::CacheSchema;

#[derive(CacheSchema)]
#[cache_schema(crate = ssa_cache_schema, crate = ssa_cache_schema)]
struct Bad {
    value: u32,
}

fn main() {}
