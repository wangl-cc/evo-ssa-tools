#![allow(
    non_upper_case_globals,
    reason = "lowercase const generic exercises derive writer identifier hygiene"
)]

use std::marker::PhantomData;

use ssa_cache_schema::{CacheSchema, schema_fingerprint};
use ssa_cache_schema as schema_alias;

#[derive(CacheSchema)]
#[cache_schema(rename = "StableName", version = "v1")]
struct Stable<'a, T, const N: usize>
where
    T: 'a,
{
    #[cache_schema(rename = "items")]
    values: [T; N],
    marker: PhantomData<&'a T>,
}

#[derive(CacheSchema)]
struct LowercaseConst<const w: usize> {
    value: [u8; w],
}

#[derive(CacheSchema)]
#[cache_schema(rename = "Event")]
enum Event<T> {
    #[cache_schema(rename = "Created")]
    Made { value: T },
    Empty,
}

#[derive(CacheSchema)]
#[serde(rename = "serde-name")]
struct SerdeAttrsAreIgnored {
    #[serde(rename = "serde_value", skip)]
    value: u32,
}

#[derive(CacheSchema)]
enum SerdeVariantAttrsAreIgnored {
    #[serde(rename = "serde_created")]
    Created {
        #[serde(rename = "serde_id")]
        id: u64,
    },
}

struct NoSchema;

#[derive(CacheSchema)]
#[cache_schema(rename = "MarkerOnly")]
struct MarkerOnly<T> {
    marker: PhantomData<T>,
}

trait HasAssoc {
    type Assoc;
}

struct Provider;

impl HasAssoc for Provider {
    type Assoc = u32;
}

#[derive(CacheSchema)]
#[cache_schema(rename = "AssocField")]
struct AssocField<T: HasAssoc> {
    value: <T as HasAssoc>::Assoc,
}

#[derive(CacheSchema)]
#[cache_schema(crate = schema_alias, rename = "CrateAlias")]
struct CrateAlias {
    value: u32,
}

fn main() {
    let _ = schema_fingerprint::<Stable<'static, u32, 4>>();
    let _ = schema_fingerprint::<LowercaseConst<4>>();
    let _ = schema_fingerprint::<Event<u32>>();
    let _ = schema_fingerprint::<SerdeAttrsAreIgnored>();
    let _ = schema_fingerprint::<SerdeVariantAttrsAreIgnored>();
    let _ = schema_fingerprint::<MarkerOnly<NoSchema>>();
    let _ = schema_fingerprint::<AssocField<Provider>>();
    let _ = schema_alias::schema_fingerprint::<CrateAlias>();
}
