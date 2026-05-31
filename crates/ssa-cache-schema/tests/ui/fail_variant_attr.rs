use ssa_cache_schema::CacheSchema;

#[derive(CacheSchema)]
enum Bad {
    #[cache_schema(version = "v1")]
    Created,
}

fn main() {}
