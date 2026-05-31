use ssa_cache_schema::CacheSchema;

#[derive(CacheSchema)]
struct Bad {
    #[serde(skip)]
    value: u32,
}

fn main() {}
