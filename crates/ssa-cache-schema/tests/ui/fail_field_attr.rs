use ssa_cache_schema::CacheSchema;

#[derive(CacheSchema)]
struct Bad {
    #[cache_schema(skip)]
    value: u32,
}

fn main() {}
