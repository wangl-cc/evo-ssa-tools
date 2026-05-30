use proc_macro::TokenStream;
use quote::{ToTokens, quote};
use syn::{
    Attribute, Data, DeriveInput, Fields, Generics, Index, LitInt, LitStr, Path, parse_macro_input,
    parse_quote,
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

fn expand_canonical_encode(input: DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let ident = input.ident;
    let attrs = input.attrs;

    let (field_tys, field_accesses, field_schema): (Vec<_>, Vec<_>, Vec<_>) = match input.data {
        Data::Struct(data) => collect_struct_fields(data.fields),
        Data::Enum(data) => {
            return Err(syn::Error::new_spanned(
                data.enum_token,
                "CanonicalEncode can only be derived for structs",
            ));
        }
        Data::Union(data) => {
            return Err(syn::Error::new_spanned(
                data.union_token,
                "CanonicalEncode can only be derived for structs",
            ));
        }
    };
    let canonical_attrs = parse_canonical_encode_attrs(&attrs)?;
    let workflow_crate = canonical_attrs.workflow_crate;
    let generics = add_trait_bounds(input.generics, &workflow_crate);
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
    let schema = format!(
        "ssa-workflow:canonical-encode:v1;version={};type={ident};fields=[{}]",
        canonical_attrs.version,
        field_schema.join(","),
    );
    let schema = syn::LitStr::new(&schema, proc_macro2::Span::call_site());

    Ok(quote! {
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
