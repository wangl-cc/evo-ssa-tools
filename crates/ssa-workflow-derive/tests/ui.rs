#[test]
fn derive_rejects_unsupported_items() {
    let tests = trybuild::TestCases::new();
    tests.compile_fail("tests/ui/cache_schema/*.rs");
    tests.compile_fail("tests/ui/canonical_encode/*.rs");
}
