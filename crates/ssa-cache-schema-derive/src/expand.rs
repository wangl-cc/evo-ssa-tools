use proc_macro2::TokenStream as TokenStream2;
use quote::{quote, quote_spanned};
use syn::{
    Data, DeriveInput, Error, Field, Fields, Result, Variant, punctuated::Punctuated,
    spanned::Spanned, token::Comma,
};

use crate::{
    attrs::{FieldAttrs, TypeAttrs, VariantAttrs},
    bounds::add_field_schema_bounds,
};

pub(crate) fn expand_cache_schema(input: DeriveInput) -> Result<TokenStream2> {
    let ident = input.ident;
    let attrs = TypeAttrs::parse(&input.attrs, &ident)?;
    let mut generics = input.generics;
    let schema_path = attrs.schema_path_tokens();
    add_field_schema_bounds(&mut generics, &input.data, &schema_path);
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
    let body = match input.data {
        Data::Struct(data) => expand_struct_body(&attrs, &schema_path, &data.fields)?,
        Data::Enum(data) => expand_enum_body(&attrs, &schema_path, &data.variants)?,
        Data::Union(data) => {
            return Err(Error::new(
                data.union_token.span(),
                "CacheSchema derive does not support unions",
            ));
        }
    };

    Ok(quote! {
        impl #impl_generics #schema_path::CacheSchema for #ident #ty_generics #where_clause {
            fn write_schema(w: &mut #schema_path::SchemaWriter) {
                #body
            }
        }
    })
}

fn expand_struct_body(
    attrs: &TypeAttrs,
    schema_path: &TokenStream2,
    fields: &Fields,
) -> Result<TokenStream2> {
    let name = attrs.name_tokens();
    let version = attrs.version_tokens();
    let field_tokens = expand_fields(fields, schema_path)?;

    Ok(quote! {
        w.struct_begin(#name);
        #version
        #field_tokens
        w.struct_end();
    })
}

fn expand_enum_body(
    attrs: &TypeAttrs,
    schema_path: &TokenStream2,
    variants: &Punctuated<Variant, Comma>,
) -> Result<TokenStream2> {
    let name = attrs.name_tokens();
    let version = attrs.version_tokens();
    let variants = variants
        .iter()
        .enumerate()
        .map(|variant| expand_variant(variant, schema_path))
        .collect::<Result<Vec<_>>>()?;

    Ok(quote! {
        w.enum_begin(#name);
        #version
        #(#variants)*
        w.enum_end();
    })
}

fn expand_variant(
    (index, variant): (usize, &Variant),
    schema_path: &TokenStream2,
) -> Result<TokenStream2> {
    let attrs = VariantAttrs::parse(&variant.attrs, &variant.ident)?;
    let name = attrs.name_tokens();
    let fields = expand_fields(&variant.fields, schema_path)?;

    Ok(quote! {
        w.variant_begin(#index, #name);
        #fields
        w.variant_end();
    })
}

fn expand_fields(fields: &Fields, schema_path: &TokenStream2) -> Result<TokenStream2> {
    fields
        .iter()
        .enumerate()
        .map(|field| expand_field(field, schema_path))
        .collect::<Result<Vec<_>>>()
        .map(|fields| quote! { #(#fields)* })
}

fn expand_field(
    (index, field): (usize, &Field),
    schema_path: &TokenStream2,
) -> Result<TokenStream2> {
    let attrs = FieldAttrs::parse(field)?;
    let name = attrs.name_tokens();
    let ty = &field.ty;

    Ok(quote_spanned! { ty.span() =>
        w.field_begin(#index, #name);
        <#ty as #schema_path::CacheSchema>::write_schema(w);
        w.field_end();
    })
}
