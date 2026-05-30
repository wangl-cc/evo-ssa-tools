use proc_macro::TokenStream;
use quote::{ToTokens, quote};
use syn::{
    Attribute, Data, DeriveInput, Fields, Generics, Index, LitInt, LitStr, Path, Type,
    TypeParamBound, Variant, meta::ParseNestedMeta, parse_macro_input, parse_quote,
};

#[proc_macro_derive(CanonicalEncode, attributes(canonical_encode))]
pub fn derive_canonical_encode(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand_canonical_encode(input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

#[proc_macro_derive(CacheSchema, attributes(cache_schema))]
pub fn derive_cache_schema(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand_cache_schema(input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

struct SchemaAttrs {
    version: u8,
    workflow_crate: Path,
}

struct EncodeAttrs {
    workflow_crate: Path,
}

enum DeriveBody {
    Struct {
        field_tys: Vec<Type>,
        field_accesses: Vec<proc_macro2::TokenStream>,
        field_schema: Vec<String>,
    },
    UnitEnum {
        variants: Vec<syn::Ident>,
        variant_schema: Vec<String>,
    },
}

struct ImplContext<'a> {
    ident: &'a syn::Ident,
    workflow_crate: &'a Path,
    impl_generics: syn::ImplGenerics<'a>,
    ty_generics: syn::TypeGenerics<'a>,
    where_clause: Option<&'a syn::WhereClause>,
}

fn expand_canonical_encode(input: DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let ident = input.ident;
    let attrs = input.attrs;
    let body = collect_derive_body(input.data, "CanonicalEncode")?;
    let encode_attrs = parse_encode_attrs(&attrs)?;
    let workflow_crate = encode_attrs.workflow_crate;
    let encode_generics = add_trait_bounds(
        input.generics,
        parse_quote!(#workflow_crate::cache::CanonicalEncode),
    );
    let (encode_impl_generics, encode_ty_generics, encode_where_clause) =
        encode_generics.split_for_impl();
    let encode_context = ImplContext {
        ident: &ident,
        workflow_crate: &workflow_crate,
        impl_generics: encode_impl_generics,
        ty_generics: encode_ty_generics,
        where_clause: encode_where_clause,
    };

    match body {
        DeriveBody::Struct {
            field_tys,
            field_accesses,
            ..
        } => Ok(expand_struct_encode_impl(
            encode_context,
            field_tys,
            field_accesses,
        )),
        DeriveBody::UnitEnum { variants, .. } => {
            Ok(expand_unit_enum_encode_impl(encode_context, variants))
        }
    }
}

fn expand_cache_schema(input: DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let ident = input.ident;
    let attrs = input.attrs;
    let body = collect_derive_body(input.data, "CacheSchema")?;
    let schema_attrs = parse_schema_attrs(&attrs, "cache_schema")?;
    let workflow_crate = schema_attrs.workflow_crate;
    let generics = add_trait_bounds(
        input.generics,
        parse_quote!(#workflow_crate::cache::CacheSchema),
    );
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
    let context = ImplContext {
        ident: &ident,
        workflow_crate: &workflow_crate,
        impl_generics,
        ty_generics,
        where_clause,
    };

    Ok(match body {
        DeriveBody::Struct {
            field_tys,
            field_schema,
            ..
        } => expand_struct_schema_impl(context, schema_attrs.version, &field_tys, &field_schema),
        DeriveBody::UnitEnum { variant_schema, .. } => {
            expand_unit_enum_schema_impl(context, schema_attrs.version, &variant_schema)
        }
    })
}

fn expand_struct_schema_impl(
    ctx: ImplContext<'_>,
    version: u8,
    field_tys: &[Type],
    field_schema: &[String],
) -> proc_macro2::TokenStream {
    let ident = ctx.ident;
    let schema = syn::LitStr::new(
        &format!(
            "ssa-workflow:cache-schema:v1;version={version};type={ident};fields=[{}]",
            field_schema.join(","),
        ),
        proc_macro2::Span::call_site(),
    );

    expand_cache_schema_impl(ctx, schema, field_tys)
}

fn expand_struct_encode_impl(
    ctx: ImplContext<'_>,
    field_tys: Vec<Type>,
    field_accesses: Vec<proc_macro2::TokenStream>,
) -> proc_macro2::TokenStream {
    let ImplContext {
        ident,
        workflow_crate,
        impl_generics,
        ty_generics,
        where_clause,
        ..
    } = ctx;

    quote! {
        unsafe impl #impl_generics #workflow_crate::cache::CanonicalEncode for #ident #ty_generics #where_clause {
            const SIZE: usize = 0 #( + <#field_tys as #workflow_crate::cache::CanonicalEncode>::SIZE )*;

            unsafe fn encode_into(&self, buffer: &mut [u8]) {
                let mut writer = #workflow_crate::cache::CanonicalEncodeWriter::for_type::<Self>(buffer);
                #( writer.write(&self.#field_accesses); )*
                writer.finish();
            }
        }
    }
}

fn expand_unit_enum_schema_impl(
    ctx: ImplContext<'_>,
    version: u8,
    variant_schema: &[String],
) -> proc_macro2::TokenStream {
    let ident = ctx.ident;
    let schema = syn::LitStr::new(
        &format!(
            "ssa-workflow:cache-schema:v1;version={version};type={ident};variants=[{}]",
            variant_schema.join(","),
        ),
        proc_macro2::Span::call_site(),
    );

    let component_tys = [parse_quote!(u8)];
    expand_cache_schema_impl(ctx, schema, &component_tys)
}

fn expand_cache_schema_impl(
    ctx: ImplContext<'_>,
    schema: LitStr,
    component_tys: &[Type],
) -> proc_macro2::TokenStream {
    let ImplContext {
        ident,
        workflow_crate,
        impl_generics,
        ty_generics,
        where_clause,
    } = ctx;

    quote! {
        unsafe impl #impl_generics #workflow_crate::cache::CacheSchema for #ident #ty_generics #where_clause {
            const SCHEMA_SIGNATURE: u32 = {
                let signature = #workflow_crate::cache::schema_signature(#schema.as_bytes());
                #(
                    let signature = #workflow_crate::cache::extend_schema_signature(
                        signature,
                        <#component_tys as #workflow_crate::cache::CacheSchema>::SCHEMA_SIGNATURE,
                    );
                )*
                signature
            };
        }
    }
}

fn expand_unit_enum_encode_impl(
    ctx: ImplContext<'_>,
    variants: Vec<syn::Ident>,
) -> proc_macro2::TokenStream {
    let ImplContext {
        ident,
        workflow_crate,
        impl_generics,
        ty_generics,
        where_clause,
        ..
    } = ctx;
    let ordinals: Vec<u8> = (0..variants.len()).map(|index| index as u8).collect();

    quote! {
        unsafe impl #impl_generics #workflow_crate::cache::CanonicalEncode for #ident #ty_generics #where_clause {
            const SIZE: usize = <u8 as #workflow_crate::cache::CanonicalEncode>::SIZE;

            unsafe fn encode_into(&self, buffer: &mut [u8]) {
                let variant = match self {
                    #( Self::#variants => #ordinals, )*
                };
                let mut writer = #workflow_crate::cache::CanonicalEncodeWriter::for_type::<Self>(buffer);
                writer.write(&variant).finish();
            }
        }
    }
}

fn parse_schema_attrs(attrs: &[Attribute], attr_name: &'static str) -> syn::Result<SchemaAttrs> {
    let mut version = None;
    let mut workflow_crate = None;
    for attr in attrs {
        if !attr.path().is_ident(attr_name) {
            continue;
        }

        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("version") {
                if version.is_some() {
                    return Err(meta.error(format!("duplicate {attr_name} version")));
                }
                let value = meta.value()?;
                let lit: LitInt = value.parse()?;
                version = Some(lit.base10_parse::<u8>()?);
                return Ok(());
            }
            if !parse_workflow_crate_meta(&meta, attr_name, &mut workflow_crate)? {
                return Err(meta.error(format!("unsupported {attr_name} attribute")));
            }
            Ok(())
        })?;
    }

    let version = version.ok_or_else(|| {
        syn::Error::new(
            proc_macro2::Span::call_site(),
            format!("missing #[{attr_name}(version = N)]"),
        )
    })?;
    let workflow_crate = workflow_crate.unwrap_or_else(|| parse_quote!(::ssa_workflow));

    Ok(SchemaAttrs {
        version,
        workflow_crate,
    })
}

fn parse_encode_attrs(attrs: &[Attribute]) -> syn::Result<EncodeAttrs> {
    let mut workflow_crate = None;
    for attr in attrs {
        if !attr.path().is_ident("canonical_encode") {
            continue;
        }

        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("version") {
                return Err(meta.error(
                    "canonical_encode no longer accepts version; use #[cache_schema(version = N)]",
                ));
            }
            if !parse_workflow_crate_meta(&meta, "canonical_encode", &mut workflow_crate)? {
                return Err(meta.error("unsupported canonical_encode attribute"));
            }
            Ok(())
        })?;
    }

    let workflow_crate = workflow_crate.unwrap_or_else(|| parse_quote!(::ssa_workflow));

    Ok(EncodeAttrs { workflow_crate })
}

fn parse_workflow_crate_meta(
    meta: &ParseNestedMeta<'_>,
    attr_name: &str,
    workflow_crate: &mut Option<Path>,
) -> syn::Result<bool> {
    if !meta.path.is_ident("crate") {
        return Ok(false);
    }
    if workflow_crate.is_some() {
        return Err(meta.error(format!("duplicate {attr_name} crate")));
    }
    let value = meta.value()?;
    let lit: LitStr = value.parse()?;
    *workflow_crate = Some(lit.parse::<Path>()?);
    Ok(true)
}

fn add_trait_bounds(mut generics: Generics, bound: TypeParamBound) -> Generics {
    for param in generics.type_params_mut() {
        param.bounds.push(bound.clone());
    }
    generics
}

fn collect_derive_body(data: Data, derive_name: &str) -> syn::Result<DeriveBody> {
    match data {
        Data::Struct(data) => Ok(collect_struct_body(data.fields)),
        Data::Enum(data) => collect_unit_enum_body(data.enum_token, data.variants, derive_name),
        Data::Union(data) => Err(unsupported_item_error(data.union_token, derive_name)),
    }
}

fn collect_struct_body(fields: Fields) -> DeriveBody {
    let (field_tys, field_accesses, field_schema) = collect_struct_fields(fields);
    DeriveBody::Struct {
        field_tys,
        field_accesses,
        field_schema,
    }
}

fn collect_unit_enum_body(
    enum_token: syn::token::Enum,
    variants: impl IntoIterator<Item = Variant>,
    derive_name: &str,
) -> syn::Result<DeriveBody> {
    let (variants, variant_schema) = collect_unit_enum_variants(variants, derive_name)?;
    if variants.is_empty() {
        return Err(syn::Error::new_spanned(
            enum_token,
            format!("{derive_name} cannot be derived for empty enums"),
        ));
    }
    if variants.len() > u8::MAX as usize + 1 {
        return Err(syn::Error::new_spanned(
            enum_token,
            format!("{derive_name} unit enums support at most 256 variants"),
        ));
    }
    Ok(DeriveBody::UnitEnum {
        variants,
        variant_schema,
    })
}

fn unsupported_item_error(token: impl ToTokens, derive_name: &str) -> syn::Error {
    syn::Error::new_spanned(
        token,
        format!("{derive_name} can only be derived for structs or unit enums"),
    )
}

fn collect_struct_fields(
    fields: Fields,
) -> (Vec<Type>, Vec<proc_macro2::TokenStream>, Vec<String>) {
    let mut field_tys = Vec::new();
    let mut field_accesses = Vec::new();
    let mut field_schema = Vec::new();

    match fields {
        Fields::Named(fields) => {
            for field in fields.named {
                let ident = field.ident.expect("named fields should have identifiers");
                let schema = format!("{}:{}", ident, type_schema(&field.ty));
                field_tys.push(field.ty);
                field_accesses.push(quote!(#ident));
                field_schema.push(schema);
            }
        }
        Fields::Unnamed(fields) => {
            for (index, field) in fields.unnamed.into_iter().enumerate() {
                let schema = format!("{index}:{}", type_schema(&field.ty));
                let index = Index::from(index);
                field_tys.push(field.ty);
                field_accesses.push(quote!(#index));
                field_schema.push(schema);
            }
        }
        Fields::Unit => {}
    }

    (field_tys, field_accesses, field_schema)
}

fn collect_unit_enum_variants(
    variants: impl IntoIterator<Item = Variant>,
    derive_name: &str,
) -> syn::Result<(Vec<syn::Ident>, Vec<String>)> {
    let mut variant_idents = Vec::new();
    let mut variant_schema = Vec::new();

    for (index, variant) in variants.into_iter().enumerate() {
        if let Some((_, discriminant)) = variant.discriminant {
            return Err(syn::Error::new_spanned(
                discriminant,
                format!(
                    "{derive_name} unit enum discriminants are not supported; variants encode by declaration order"
                ),
            ));
        }
        if !matches!(variant.fields, Fields::Unit) {
            return Err(syn::Error::new_spanned(
                variant.ident,
                format!("{derive_name} can only be derived for unit enum variants"),
            ));
        }
        variant_schema.push(format!("{index}:{}", variant.ident));
        variant_idents.push(variant.ident);
    }

    Ok((variant_idents, variant_schema))
}

fn type_schema(ty: &Type) -> String {
    ty.to_token_stream().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_attrs_default_to_standard_workflow_crate_path() {
        let attrs: Vec<Attribute> = Vec::new();

        let parsed = parse_encode_attrs(&attrs).unwrap();

        assert_eq!(
            parsed.workflow_crate.to_token_stream().to_string(),
            ":: ssa_workflow"
        );
    }

    #[test]
    fn encode_attrs_can_override_workflow_crate_path() {
        let attrs: Vec<Attribute> = vec![parse_quote!(#[canonical_encode(crate = "workflow")])];

        let parsed = parse_encode_attrs(&attrs).unwrap();

        assert_eq!(
            parsed.workflow_crate.to_token_stream().to_string(),
            "workflow"
        );
    }

    #[test]
    fn cache_schema_attrs_use_cache_schema_attribute() {
        let attrs: Vec<Attribute> =
            vec![parse_quote!(#[cache_schema(version = 7, crate = "workflow")])];

        let parsed = parse_schema_attrs(&attrs, "cache_schema").unwrap();

        assert_eq!(parsed.version, 7);
        assert_eq!(
            parsed.workflow_crate.to_token_stream().to_string(),
            "workflow"
        );
    }
}
