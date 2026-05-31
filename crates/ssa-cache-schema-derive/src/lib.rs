use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{quote, quote_spanned};
use syn::{
    Attribute, Data, DeriveInput, Error, Field, Fields, Generics, Ident, LitStr, Path, Result,
    Type, Variant, parse_macro_input, parse_quote, spanned::Spanned,
};

#[proc_macro_derive(CacheSchema, attributes(cache_schema, serde))]
pub fn derive_cache_schema(input: TokenStream) -> TokenStream {
    match expand_cache_schema(parse_macro_input!(input as DeriveInput)) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.to_compile_error().into(),
    }
}

fn expand_cache_schema(input: DeriveInput) -> Result<TokenStream2> {
    reject_serde_attrs(&input.attrs)?;

    let ident = input.ident;
    let attrs = TypeAttrs::parse(&input.attrs, &ident)?;
    let mut generics = input.generics;
    let schema_path = attrs.schema_path_tokens();
    add_field_schema_bounds(&mut generics, &input.data, &schema_path);
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
    let body = match input.data {
        Data::Struct(data) => expand_struct_body(&attrs, &schema_path, &data.fields)?,
        Data::Enum(data) => expand_enum_body(
            &attrs,
            &schema_path,
            &data.variants.into_iter().collect::<Vec<_>>(),
        )?,
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
    for field in fields {
        reject_serde_attrs(&field.attrs)?;
    }

    let module = attrs.module_tokens();
    let name = attrs.name_tokens();
    let version = attrs.version_tokens();
    let field_tokens = expand_fields(fields, schema_path)?;

    Ok(quote! {
        w.struct_begin(#module, #name);
        #version
        #field_tokens
        w.struct_end();
    })
}

fn expand_enum_body(
    attrs: &TypeAttrs,
    schema_path: &TokenStream2,
    variants: &[Variant],
) -> Result<TokenStream2> {
    for variant in variants {
        reject_serde_attrs(&variant.attrs)?;
        for field in &variant.fields {
            reject_serde_attrs(&field.attrs)?;
        }
    }

    let module = attrs.module_tokens();
    let name = attrs.name_tokens();
    let version = attrs.version_tokens();
    let variants = variants
        .iter()
        .enumerate()
        .map(|variant| expand_variant(variant, schema_path))
        .collect::<Result<Vec<_>>>()?;

    Ok(quote! {
        w.enum_begin(#module, #name);
        #version
        #(#variants)*
        w.enum_end();
    })
}

fn expand_variant(
    (index, variant): (usize, &Variant),
    schema_path: &TokenStream2,
) -> Result<TokenStream2> {
    let attrs = ItemAttrs::parse(&variant.attrs, &variant.ident)?;
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

fn add_field_schema_bounds(generics: &mut Generics, data: &Data, schema_path: &TokenStream2) {
    let field_types = schema_field_types(data);
    if field_types.is_empty() {
        return;
    }

    let where_clause = generics.make_where_clause();
    for ty in field_types {
        where_clause
            .predicates
            .push(parse_quote!(#ty: #schema_path::CacheSchema));
    }
}

fn schema_field_types(data: &Data) -> Vec<&Type> {
    match data {
        Data::Struct(data) => data.fields.iter().map(|field| &field.ty).collect(),
        Data::Enum(data) => data
            .variants
            .iter()
            .flat_map(|variant| variant.fields.iter().map(|field| &field.ty))
            .collect(),
        Data::Union(_) => Vec::new(),
    }
}

fn set_once<T>(slot: &mut Option<T>, value: T, attr: &str, span: proc_macro2::Span) -> Result<()> {
    if slot.is_some() {
        Err(Error::new(
            span,
            format!("duplicate cache_schema {attr} attribute"),
        ))
    } else {
        *slot = Some(value);
        Ok(())
    }
}

fn parse_lit_str_value(
    slot: &mut Option<LitStr>,
    meta: syn::meta::ParseNestedMeta<'_>,
    attr: &str,
) -> Result<()> {
    let span = meta.path.span();
    let value = meta.value()?.parse()?;
    set_once(slot, value, attr, span)
}

fn parse_path_value(
    slot: &mut Option<Path>,
    meta: syn::meta::ParseNestedMeta<'_>,
    attr: &str,
) -> Result<()> {
    let span = meta.path.span();
    let value = meta.value()?.parse()?;
    set_once(slot, value, attr, span)
}

fn default_schema_path() -> TokenStream2 {
    quote! { ::ssa_cache_schema }
}

fn quote_path(path: &Path) -> TokenStream2 {
    quote! { #path }
}

fn is_crate_attr(path: &Path) -> bool {
    path.segments.len() == 1
        && path
            .segments
            .first()
            .is_some_and(|segment| segment.ident == "crate")
}

fn reject_serde_attrs(attrs: &[Attribute]) -> Result<()> {
    for attr in attrs {
        if attr.path().is_ident("serde") {
            return Err(Error::new_spanned(
                attr,
                "CacheSchema derive does not support serde attributes; use cache_schema attributes or write CacheSchema manually",
            ));
        }
    }
    Ok(())
}

#[derive(Default)]
struct TypeAttrs {
    rename: Option<LitStr>,
    module: Option<LitStr>,
    version: Option<LitStr>,
    schema_crate: Option<Path>,
}

impl TypeAttrs {
    fn parse(attrs: &[Attribute], ident: &Ident) -> Result<Self> {
        let mut parsed = Self::default();
        for attr in attrs {
            if !attr.path().is_ident("cache_schema") {
                continue;
            }
            attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("rename") {
                    parse_lit_str_value(&mut parsed.rename, meta, "rename")
                } else if meta.path.is_ident("module") {
                    parse_lit_str_value(&mut parsed.module, meta, "module")
                } else if meta.path.is_ident("version") {
                    parse_lit_str_value(&mut parsed.version, meta, "version")
                } else if is_crate_attr(&meta.path) {
                    parse_path_value(&mut parsed.schema_crate, meta, "crate")
                } else {
                    Err(meta.error("unsupported cache_schema type attribute"))
                }
            })?;
        }
        if parsed.rename.is_none() {
            parsed.rename = Some(LitStr::new(&ident.to_string(), ident.span()));
        }
        Ok(parsed)
    }

    fn schema_path_tokens(&self) -> TokenStream2 {
        match &self.schema_crate {
            Some(schema_crate) => quote_path(schema_crate),
            None => default_schema_path(),
        }
    }

    fn module_tokens(&self) -> TokenStream2 {
        match &self.module {
            Some(module) => quote! { #module },
            None => quote! { module_path!() },
        }
    }

    fn name_tokens(&self) -> TokenStream2 {
        let rename = self
            .rename
            .as_ref()
            .expect("TypeAttrs::parse always fills rename");
        quote! { #rename }
    }

    fn version_tokens(&self) -> TokenStream2 {
        match &self.version {
            Some(version) => quote! { w.type_version(#version); },
            None => TokenStream2::new(),
        }
    }
}

#[derive(Default)]
struct ItemAttrs {
    rename: Option<LitStr>,
}

impl ItemAttrs {
    fn parse(attrs: &[Attribute], ident: &Ident) -> Result<Self> {
        let mut parsed = Self::default();
        for attr in attrs {
            if !attr.path().is_ident("cache_schema") {
                continue;
            }
            attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("rename") {
                    parse_lit_str_value(&mut parsed.rename, meta, "rename")
                } else {
                    Err(meta.error("unsupported cache_schema variant attribute"))
                }
            })?;
        }
        if parsed.rename.is_none() {
            parsed.rename = Some(LitStr::new(&ident.to_string(), ident.span()));
        }
        Ok(parsed)
    }

    fn name_tokens(&self) -> TokenStream2 {
        let rename = self
            .rename
            .as_ref()
            .expect("ItemAttrs::parse always fills rename");
        quote! { #rename }
    }
}

#[derive(Default)]
struct FieldAttrs {
    rename: Option<LitStr>,
}

impl FieldAttrs {
    fn parse(field: &Field) -> Result<Self> {
        let mut parsed = Self::default();
        for attr in &field.attrs {
            if !attr.path().is_ident("cache_schema") {
                continue;
            }
            attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("rename") {
                    parse_lit_str_value(&mut parsed.rename, meta, "rename")
                } else {
                    Err(meta.error("unsupported cache_schema field attribute"))
                }
            })?;
        }
        if parsed.rename.is_none()
            && let Some(ident) = &field.ident
        {
            parsed.rename = Some(LitStr::new(&ident.to_string(), ident.span()));
        }
        Ok(parsed)
    }

    fn name_tokens(&self) -> TokenStream2 {
        match &self.rename {
            Some(rename) => quote! { Some(#rename) },
            None => quote! { None },
        }
    }
}
