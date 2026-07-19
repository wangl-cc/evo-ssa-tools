use ssa_cache_schema::CacheSchema;

#[derive(CacheSchema)]
struct Node {
    next: Option<Box<Node>>,
}

fn main() {}
