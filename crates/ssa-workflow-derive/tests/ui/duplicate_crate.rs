use ssa_workflow_derive::CanonicalEncode;

#[derive(CanonicalEncode)]
#[canonical_encode(version = 1, crate = "workflow", crate = "other_workflow")]
struct DuplicateCrate {
    value: u64,
}

fn main() {}
