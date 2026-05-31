use ssa_cache_schema::CacheSchema;

#[derive(CacheSchema)]
union Bad {
    value: u32,
}

fn main() {}
