#![allow(
    dead_code,
    reason = "derive fixtures are fingerprinted by type and intentionally never constructed"
)]

use std::marker::PhantomData;

use ssa_cache_schema::{CacheSchema, schema_fingerprint};

#[path = "derive/derive_attrs.rs"]
mod derive_attrs;
#[path = "derive/derive_enums.rs"]
mod derive_enums;
#[path = "derive/derive_generics.rs"]
mod derive_generics;
#[path = "derive/derive_golden.rs"]
mod derive_golden;
#[path = "derive/derive_names.rs"]
mod derive_names;
#[path = "derive/derive_structs.rs"]
mod derive_structs;
