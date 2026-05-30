use ssa_workflow_derive::CanonicalEncode;

#[derive(CanonicalEncode)]
#[canonical_encode(version = 1)]
struct VersionOnCanonicalEncode {
    value: u64,
}

fn main() {}
