use ssa_workflow_derive::CanonicalEncode;

#[derive(CanonicalEncode)]
struct MissingVersion {
    value: u64,
}

fn main() {}
