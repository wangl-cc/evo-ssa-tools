use proc_macro::TokenStream;
use quote::{ToTokens, quote};
use syn::{
    Attribute, Data, DeriveInput, Fields, Generics, Index, LitInt, LitStr, Path, Variant,
    parse_macro_input, parse_quote,
};

#[proc_macro_derive(CanonicalEncode, attributes(canonical_encode))]
pub fn derive_canonical_encode(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand_canonical_encode(input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

struct CanonicalEncodeAttrs {
    version: u8,
    workflow_crate: Path,
}

enum EncodeBody {
    Struct {
        field_tys: Vec<syn::Type>,
        field_accesses: Vec<proc_macro2::TokenStream>,
        schema_parts: Vec<String>,
    },
    UnitEnum {
        variants: Vec<syn::Ident>,
        schema_parts: Vec<String>,
    },
}

fn expand_canonical_encode(input: DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let ident = input.ident;
    let attrs = input.attrs;
    let encode_body = match input.data {
        Data::Struct(data) => collect_struct_body(data.fields),
        Data::Enum(data) => collect_unit_enum_body(data.enum_token, data.variants)?,
        Data::Union(data) => {
            return Err(syn::Error::new_spanned(
                data.union_token,
                "CanonicalEncode can only be derived for structs or unit enums",
            ));
        }
    };
    let canonical_attrs = parse_canonical_encode_attrs(&attrs)?;
    let workflow_crate = canonical_attrs.workflow_crate;
    let generics = add_trait_bounds(input.generics, &workflow_crate);
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    Ok(match encode_body {
        EncodeBody::Struct {
            field_tys,
            field_accesses,
            schema_parts,
        } => {
            let schema = format!(
                "ssa-workflow:canonical-encode:v1;version={};type={ident};fields=[{}]",
                canonical_attrs.version,
                schema_parts.join(","),
            );
            let schema = syn::LitStr::new(&schema, proc_macro2::Span::call_site());
            quote! {
                unsafe impl #impl_generics #workflow_crate::cache::CanonicalEncode for #ident #ty_generics #where_clause {
                    const SIZE: usize = 0 #( + <#field_tys as #workflow_crate::cache::CanonicalEncode>::SIZE )*;
                    const SCHEMA_SIGNATURE: u32 = {
                        let signature = #workflow_crate::cache::schema_signature(#schema.as_bytes());
                        #(
                            let signature = #workflow_crate::cache::extend_schema_signature(
                                signature,
                                <#field_tys as #workflow_crate::cache::CanonicalEncode>::SCHEMA_SIGNATURE,
                            );
                        )*
                        signature
                    };

                    unsafe fn encode_into(&self, buffer: &mut [u8]) {
                        let mut writer = #workflow_crate::cache::CanonicalEncodeWriter::for_type::<Self>(buffer);
                        #( writer.write(&self.#field_accesses); )*
                        writer.finish();
                    }
                }
            }
        }
        EncodeBody::UnitEnum {
            variants,
            schema_parts,
        } => {
            let schema = format!(
                "ssa-workflow:canonical-encode:v1;version={};type={ident};variants=[{}]",
                canonical_attrs.version,
                schema_parts.join(","),
            );
            let schema = syn::LitStr::new(&schema, proc_macro2::Span::call_site());
            let ordinals: Vec<u8> = (0..variants.len()).map(|index| index as u8).collect();
            quote! {
                unsafe impl #impl_generics #workflow_crate::cache::CanonicalEncode for #ident #ty_generics #where_clause {
                    const SIZE: usize = <u8 as #workflow_crate::cache::CanonicalEncode>::SIZE;
                    const SCHEMA_SIGNATURE: u32 = {
                        let signature = #workflow_crate::cache::schema_signature(#schema.as_bytes());
                        #workflow_crate::cache::extend_schema_signature(
                            signature,
                            <u8 as #workflow_crate::cache::CanonicalEncode>::SCHEMA_SIGNATURE,
                        )
                    };

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
    })
}

fn parse_canonical_encode_attrs(attrs: &[Attribute]) -> syn::Result<CanonicalEncodeAttrs> {
    let mut version = None;
    let mut workflow_crate = None;
    for attr in attrs {
        if !attr.path().is_ident("canonical_encode") {
            continue;
        }

        attr.parse_nested_meta(|meta| {
            if !meta.path.is_ident("version") {
                if meta.path.is_ident("crate") {
                    if workflow_crate.is_some() {
                        return Err(meta.error("duplicate canonical_encode crate"));
                    }
                    let value = meta.value()?;
                    let lit: LitStr = value.parse()?;
                    workflow_crate = Some(lit.parse::<Path>()?);
                    return Ok(());
                }

                return Err(meta.error("unsupported canonical_encode attribute"));
            }
            if version.is_some() {
                return Err(meta.error("duplicate canonical_encode version"));
            }
            let value = meta.value()?;
            let lit: LitInt = value.parse()?;
            version = Some(lit.base10_parse::<u8>()?);
            Ok(())
        })?;
    }

    let version = version.ok_or_else(|| {
        syn::Error::new(
            proc_macro2::Span::call_site(),
            "missing #[canonical_encode(version = N)]",
        )
    })?;
    let workflow_crate = workflow_crate.unwrap_or_else(|| parse_quote!(::ssa_workflow));

    Ok(CanonicalEncodeAttrs {
        version,
        workflow_crate,
    })
}

fn add_trait_bounds(mut generics: Generics, workflow_crate: &Path) -> Generics {
    for param in generics.type_params_mut() {
        param.bounds.push(parse_quote!(
            #workflow_crate::cache::CanonicalEncode
        ));
    }
    generics
}

fn collect_struct_body(fields: Fields) -> EncodeBody {
    let (field_tys, field_accesses, schema_parts) = collect_struct_fields(fields);
    EncodeBody::Struct {
        field_tys,
        field_accesses,
        schema_parts,
    }
}

fn collect_unit_enum_body(
    enum_token: syn::token::Enum,
    variants: impl IntoIterator<Item = Variant>,
) -> syn::Result<EncodeBody> {
    let (variants, schema_parts) = collect_unit_enum_variants(variants)?;
    if variants.is_empty() {
        return Err(syn::Error::new_spanned(
            enum_token,
            "CanonicalEncode cannot be derived for empty enums",
        ));
    }
    if variants.len() > u8::MAX as usize + 1 {
        return Err(syn::Error::new_spanned(
            enum_token,
            "CanonicalEncode unit enums support at most 256 variants",
        ));
    }
    Ok(EncodeBody::UnitEnum {
        variants,
        schema_parts,
    })
}

fn collect_struct_fields(
    fields: Fields,
) -> (Vec<syn::Type>, Vec<proc_macro2::TokenStream>, Vec<String>) {
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
) -> syn::Result<(Vec<syn::Ident>, Vec<String>)> {
    let mut variant_idents = Vec::new();
    let mut variant_schema = Vec::new();

    for (index, variant) in variants.into_iter().enumerate() {
        if let Some((_, discriminant)) = variant.discriminant {
            return Err(syn::Error::new_spanned(
                discriminant,
                "CanonicalEncode unit enum discriminants are not supported; variants encode by declaration order",
            ));
        }
        if !matches!(variant.fields, Fields::Unit) {
            return Err(syn::Error::new_spanned(
                variant.ident,
                "CanonicalEncode can only be derived for unit enum variants",
            ));
        }
        variant_schema.push(format!("{index}:{}", variant.ident));
        variant_idents.push(variant.ident);
    }

    Ok((variant_idents, variant_schema))
}

fn type_schema(ty: &syn::Type) -> String {
    ty.to_token_stream().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attrs_default_to_standard_workflow_crate_path() {
        let attrs: Vec<Attribute> = vec![parse_quote!(#[canonical_encode(version = 7)])];

        let parsed = parse_canonical_encode_attrs(&attrs).unwrap();

        assert_eq!(parsed.version, 7);
        assert_eq!(
            parsed.workflow_crate.to_token_stream().to_string(),
            ":: ssa_workflow"
        );
    }

    #[test]
    fn attrs_can_override_workflow_crate_path() {
        let attrs: Vec<Attribute> =
            vec![parse_quote!(#[canonical_encode(version = 7, crate = "workflow")])];

        let parsed = parse_canonical_encode_attrs(&attrs).unwrap();

        assert_eq!(parsed.version, 7);
        assert_eq!(
            parsed.workflow_crate.to_token_stream().to_string(),
            "workflow"
        );
    }
}
