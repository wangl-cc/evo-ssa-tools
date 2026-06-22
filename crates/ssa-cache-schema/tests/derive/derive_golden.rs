use super::{CacheSchema, schema_fingerprint};

#[test]
fn schema_fingerprints_match_golden_vectors() {
    #[derive(CacheSchema)]
    #[cache_schema(rename = "GoldenStruct")]
    struct GoldenStruct {
        width: u32,
        heights: Vec<u16>,
    }

    #[derive(CacheSchema)]
    #[cache_schema(rename = "GoldenEnum")]
    enum GoldenEnum {
        Empty,
        Value { id: u64 },
    }

    #[derive(CacheSchema)]
    #[cache_schema(rename = "GoldenVersioned", version = "v1")]
    struct GoldenVersioned {
        value: u8,
    }

    #[derive(CacheSchema)]
    #[cache_schema(rename = "GoldenEmpty")]
    struct GoldenUnit;

    assert_eq!(schema_fingerprint::<u32>(), [
        176, 250, 45, 168, 40, 240, 228, 73, 44, 176, 60, 161, 26, 120, 162, 78
    ]);
    assert_eq!(schema_fingerprint::<Vec<u32>>(), [
        22, 240, 44, 117, 234, 56, 150, 21, 192, 237, 101, 197, 132, 69, 134, 41
    ]);
    assert_eq!(schema_fingerprint::<GoldenStruct>(), [
        122, 179, 166, 139, 48, 98, 88, 125, 104, 38, 59, 116, 5, 10, 227, 57
    ]);
    assert_eq!(schema_fingerprint::<GoldenEnum>(), [
        113, 194, 83, 155, 51, 236, 0, 224, 175, 218, 48, 145, 183, 219, 23, 33
    ]);
    assert_eq!(schema_fingerprint::<GoldenVersioned>(), [
        211, 152, 5, 168, 201, 103, 34, 197, 12, 214, 218, 61, 66, 107, 107, 80
    ]);
    assert_eq!(schema_fingerprint::<GoldenUnit>(), [
        1, 103, 239, 86, 59, 158, 61, 140, 198, 139, 114, 163, 78, 184, 107, 114
    ]);
}
