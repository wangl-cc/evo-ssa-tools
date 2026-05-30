use ssa_workflow_derive::CanonicalEncode;

#[derive(CanonicalEncode)]
#[canonical_encode(version = 1, version = 2)]
struct DuplicateVersion {
    value: u64,
}

fn main() {}
