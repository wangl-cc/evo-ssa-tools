use ssa_workflow_derive::CanonicalEncode;

#[derive(CanonicalEncode)]
enum BadEnum {
    A(u8),
}

fn main() {}
