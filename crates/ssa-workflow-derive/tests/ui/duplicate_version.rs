use ssa_workflow_derive::CacheSchema;

#[derive(CacheSchema)]
#[cache_schema(version = 1, version = 2)]
struct DuplicateVersion {
    value: u64,
}

fn main() {}
