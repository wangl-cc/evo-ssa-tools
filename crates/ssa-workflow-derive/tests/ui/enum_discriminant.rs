use ssa_workflow_derive::CanonicalEncode;

#[derive(CanonicalEncode)]
enum BadEnum {
    A = 1,
    B,
}

fn main() {}
