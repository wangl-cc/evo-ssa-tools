use ssa_workflow_derive::CanonicalEncode;

#[derive(CanonicalEncode)]
union BadUnion {
    value: u64,
}

fn main() {}
