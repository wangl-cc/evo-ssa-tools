use ssa_workflow_derive::CanonicalEncode;

#[derive(CanonicalEncode)]
#[canonical_encode(version = 1)]
enum BadEnum {
    A = 1,
    B,
}

fn main() {}
