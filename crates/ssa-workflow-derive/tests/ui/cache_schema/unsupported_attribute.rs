use ssa_workflow_derive::CacheSchema;

#[derive(CacheSchema)]
#[cache_schema(foo)]
struct UnsupportedAttribute {
    value: u64,
}

fn main() {}
