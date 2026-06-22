use proc_macro2::{Ident, Span, TokenStream as TokenStream2};
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
    let writer = Ident::new("__cache_schema_writer", Span::mixed_site());
    let body = match input.data {
        Data::Struct(data) => expand_struct_body(&attrs, &schema_path, &writer, &data.fields)?,
        Data::Enum(data) => expand_enum_body(&attrs, &schema_path, &writer, &data.variants)?,
        Data::Union(data) => {
            return Err(Error::new(
                data.union_token.span(),
                "CacheSchema derive does not support unions",
            ));
        }
    };

    Ok(quote! {
        impl #impl_generics #schema_path::CacheSchema for #ident #ty_generics #where_clause {
            fn write_schema(#writer: &mut #schema_path::SchemaWriter) {
                #body
            }
        }
    })
}

fn expand_struct_body(
    attrs: &TypeAttrs,
    schema_path: &TokenStream2,
    writer: &Ident,
    fields: &Fields,
) -> Result<TokenStream2> {
    let name = attrs.name_tokens();
    let version = attrs.version_tokens(writer);
    let style = expand_empty_product_style(fields, schema_path, writer);
    let field_tokens = expand_fields(fields, schema_path, writer)?;

    Ok(quote! {
        #writer.struct_begin(#name);
        #version
        #style
        #field_tokens
        #writer.struct_end();
    })
}

fn expand_enum_body(
    attrs: &TypeAttrs,
    schema_path: &TokenStream2,
    writer: &Ident,
    variants: &Punctuated<Variant, Comma>,
) -> Result<TokenStream2> {
    let name = attrs.name_tokens();
    let version = attrs.version_tokens(writer);
    let variants = variants
        .iter()
        .enumerate()
        .map(|variant| expand_variant(variant, schema_path, writer))
        .collect::<Result<Vec<_>>>()?;

    Ok(quote! {
        #writer.enum_begin(#name);
        #version
        #(#variants)*
        #writer.enum_end();
    })
}

fn expand_variant(
    (index, variant): (usize, &Variant),
    schema_path: &TokenStream2,
    writer: &Ident,
) -> Result<TokenStream2> {
    let attrs = VariantAttrs::parse(&variant.attrs, &variant.ident)?;
    let name = attrs.name_tokens();
    let style = expand_empty_product_style(&variant.fields, schema_path, writer);
    let fields = expand_fields(&variant.fields, schema_path, writer)?;

    Ok(quote! {
        #writer.variant_begin(#index, #name);
        #style
        #fields
        #writer.variant_end();
    })
}

fn expand_empty_product_style(
    fields: &Fields,
    schema_path: &TokenStream2,
    writer: &Ident,
) -> TokenStream2 {
    let style = match fields {
        Fields::Unit => Some(quote! { #schema_path::EmptyProductStyle::Unit }),
        Fields::Unnamed(fields) if fields.unnamed.is_empty() => {
            Some(quote! { #schema_path::EmptyProductStyle::Tuple })
        }
        Fields::Named(fields) if fields.named.is_empty() => {
            Some(quote! { #schema_path::EmptyProductStyle::Named })
        }
        Fields::Named(_) | Fields::Unnamed(_) => None,
    };

    match style {
        Some(style) => quote! { #writer.empty_product_style(#style); },
        None => TokenStream2::new(),
    }
}

fn expand_fields(
    fields: &Fields,
    schema_path: &TokenStream2,
    writer: &Ident,
) -> Result<TokenStream2> {
    fields
        .iter()
        .enumerate()
        .map(|field| expand_field(field, schema_path, writer))
        .collect::<Result<Vec<_>>>()
        .map(|fields| quote! { #(#fields)* })
}

fn expand_field(
    (index, field): (usize, &Field),
    schema_path: &TokenStream2,
    writer: &Ident,
) -> Result<TokenStream2> {
    let attrs = FieldAttrs::parse(field)?;
    let name = attrs.name_tokens();
    let ty = &field.ty;

    Ok(quote_spanned! { ty.span() =>
        #writer.field_begin(#index, #name);
        <#ty as #schema_path::CacheSchema>::write_schema(#writer);
        #writer.field_end();
    })
}
