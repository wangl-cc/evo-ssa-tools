#[test]
fn derive_rejects_unsupported_items() {
    let tests = trybuild::TestCases::new();
    tests.compile_fail("tests/ui/*.rs");
}
