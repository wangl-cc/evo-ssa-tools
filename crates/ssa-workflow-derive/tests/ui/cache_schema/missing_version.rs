use ssa_workflow_derive::CacheSchema;

#[derive(CacheSchema)]
struct MissingVersion {
    value: u64,
}

fn main() {}
