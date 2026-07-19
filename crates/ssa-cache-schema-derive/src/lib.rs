use proc_macro::TokenStream;
use syn::{DeriveInput, parse_macro_input};

mod attrs;
mod bounds;
mod expand;

#[proc_macro_derive(CacheSchema, attributes(cache_schema, serde))]
pub fn derive_cache_schema(input: TokenStream) -> TokenStream {
    match expand::expand_cache_schema(parse_macro_input!(input as DeriveInput)) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.to_compile_error().into(),
    }
}
