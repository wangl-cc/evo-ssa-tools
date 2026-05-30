use ssa_workflow_derive::CanonicalEncode;

#[derive(CanonicalEncode)]
#[canonical_encode(foo)]
struct UnsupportedAttribute {
    value: u64,
}

fn main() {}
