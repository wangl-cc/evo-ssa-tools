#[test]
fn derive_compile_contract() {
    let cases = trybuild::TestCases::new();
    cases.pass("tests/ui/pass_derive_contract.rs");
    cases.compile_fail("tests/ui/fail_*.rs");
}
