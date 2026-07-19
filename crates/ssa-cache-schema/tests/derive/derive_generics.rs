use super::{CacheSchema, PhantomData, schema_fingerprint};

#[test]
fn supports_tuple_unit_generic_and_phantom_shapes() {
    #[derive(CacheSchema)]
    #[cache_schema(rename = "Tuple")]
    struct Tuple<T>(T, String);

    #[derive(CacheSchema)]
    #[cache_schema(rename = "Unit")]
    struct Unit;

    #[derive(CacheSchema)]
    #[cache_schema(rename = "WithPhantom")]
    struct WithPhantom<T> {
        value: u32,
        marker: PhantomData<T>,
    }

    let tuple = schema_fingerprint::<Tuple<u32>>();
    let unit = schema_fingerprint::<Unit>();
    let phantom_u32 = schema_fingerprint::<WithPhantom<u32>>();
    let phantom_u64 = schema_fingerprint::<WithPhantom<u64>>();

    assert_ne!(tuple, unit);
    assert_eq!(phantom_u32, phantom_u64);
}

#[test]
fn supports_lifetime_const_generics_and_where_clauses() {
    #[derive(CacheSchema)]
    #[cache_schema(rename = "BorrowedArray")]
    struct BorrowedArray<'a, T, const N: usize>
    where
        T: 'a,
    {
        values: [T; N],
        marker: PhantomData<&'a T>,
    }

    assert_ne!(
        schema_fingerprint::<BorrowedArray<'static, u32, 2>>(),
        schema_fingerprint::<BorrowedArray<'static, u32, 3>>()
    );
    assert_ne!(
        schema_fingerprint::<BorrowedArray<'static, u32, 2>>(),
        schema_fingerprint::<BorrowedArray<'static, u64, 2>>()
    );
}
