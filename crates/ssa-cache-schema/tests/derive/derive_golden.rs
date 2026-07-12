use std::collections::{BTreeMap, BTreeSet};

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

    #[derive(CacheSchema)]
    #[cache_schema(rename = "GoldenEmpty")]
    struct GoldenTuple();

    #[derive(CacheSchema)]
    #[cache_schema(rename = "GoldenEmpty")]
    struct GoldenNamed {}

    assert_eq!(schema_fingerprint::<u32>(), [
        176, 250, 45, 168, 40, 240, 228, 73, 44, 176, 60, 161, 26, 120, 162, 78
    ]);
    assert_eq!(schema_fingerprint::<()>(), [
        144, 117, 177, 222, 227, 22, 74, 87, 224, 239, 117, 26, 192, 208, 10, 249
    ]);
    assert_eq!(schema_fingerprint::<(u8, u16)>(), [
        63, 19, 158, 241, 71, 105, 196, 113, 4, 20, 203, 96, 107, 154, 155, 79
    ]);
    assert_eq!(schema_fingerprint::<[u8; 3]>(), [
        20, 152, 201, 13, 154, 19, 116, 100, 98, 250, 168, 108, 110, 84, 104, 19
    ]);
    assert_eq!(schema_fingerprint::<String>(), [
        203, 150, 128, 146, 63, 211, 110, 18, 98, 130, 109, 191, 184, 169, 224, 108
    ]);
    assert_eq!(schema_fingerprint::<Vec<u32>>(), [
        22, 240, 44, 117, 234, 56, 150, 21, 192, 237, 101, 197, 132, 69, 134, 41
    ]);
    assert_eq!(schema_fingerprint::<BTreeMap<u8, u16>>(), [
        16, 192, 88, 144, 171, 5, 99, 106, 69, 248, 81, 160, 78, 191, 184, 97
    ]);
    assert_eq!(schema_fingerprint::<BTreeSet<u8>>(), [
        207, 22, 221, 202, 2, 210, 83, 175, 28, 217, 186, 64, 132, 161, 21, 35
    ]);
    assert_eq!(schema_fingerprint::<Option<u8>>(), [
        204, 202, 41, 145, 73, 153, 9, 26, 114, 111, 214, 183, 232, 75, 28, 87
    ]);
    assert_eq!(schema_fingerprint::<Result<u8, u16>>(), [
        42, 181, 122, 112, 193, 27, 143, 218, 32, 68, 188, 1, 37, 200, 24, 78
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
    assert_eq!(schema_fingerprint::<GoldenTuple>(), [
        221, 50, 192, 81, 212, 75, 197, 242, 209, 49, 195, 38, 151, 22, 146, 46
    ]);
    assert_eq!(schema_fingerprint::<GoldenNamed>(), [
        28, 189, 159, 129, 45, 210, 227, 21, 199, 2, 59, 90, 187, 221, 28, 205
    ]);
}
