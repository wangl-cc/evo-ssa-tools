use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::quote;
use syn::{
    Attribute, Error, Field, Ident, LitStr, Path, Result, meta::ParseNestedMeta, spanned::Spanned,
};

#[derive(Default)]
pub(crate) struct TypeAttrs {
    rename: Option<LitStr>,
    version: Option<LitStr>,
    schema_crate: Option<Path>,
}

impl TypeAttrs {
    pub(crate) fn parse(attrs: &[Attribute], ident: &Ident) -> Result<Self> {
        let mut parsed = Self::default();
        for attr in attrs {
            if !attr.path().is_ident("cache_schema") {
                continue;
            }
            attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("rename") {
                    parse_lit_str_value(&mut parsed.rename, meta, "rename")
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
            parsed.rename = Some(default_ident_name(ident));
        }
        Ok(parsed)
    }

    pub(crate) fn schema_path_tokens(&self) -> TokenStream2 {
        match &self.schema_crate {
            Some(schema_crate) => quote! { #schema_crate },
            None => quote! { ::ssa_cache_schema },
        }
    }

    pub(crate) fn name_tokens(&self) -> TokenStream2 {
        let rename = self
            .rename
            .as_ref()
            .expect("TypeAttrs::parse always fills rename");
        quote! { #rename }
    }

    pub(crate) fn version_tokens(&self) -> TokenStream2 {
        match &self.version {
            Some(version) => quote! { w.type_version(#version); },
            None => TokenStream2::new(),
        }
    }
}

#[derive(Default)]
pub(crate) struct VariantAttrs {
    rename: Option<LitStr>,
}

impl VariantAttrs {
    pub(crate) fn parse(attrs: &[Attribute], ident: &Ident) -> Result<Self> {
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
            parsed.rename = Some(default_ident_name(ident));
        }
        Ok(parsed)
    }

    pub(crate) fn name_tokens(&self) -> TokenStream2 {
        let rename = self
            .rename
            .as_ref()
            .expect("VariantAttrs::parse always fills rename");
        quote! { #rename }
    }
}

#[derive(Default)]
pub(crate) struct FieldAttrs {
    rename: Option<LitStr>,
}

impl FieldAttrs {
    pub(crate) fn parse(field: &Field) -> Result<Self> {
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
            parsed.rename = Some(default_ident_name(ident));
        }
        Ok(parsed)
    }

    pub(crate) fn name_tokens(&self) -> TokenStream2 {
        match &self.rename {
            Some(rename) => quote! { Some(#rename) },
            None => quote! { None },
        }
    }
}

fn set_once<T>(slot: &mut Option<T>, value: T, attr: &str, span: Span) -> Result<()> {
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
    meta: ParseNestedMeta<'_>,
    attr: &str,
) -> Result<()> {
    let span = meta.path.span();
    let value = meta.value()?.parse()?;
    set_once(slot, value, attr, span)
}

fn parse_path_value(slot: &mut Option<Path>, meta: ParseNestedMeta<'_>, attr: &str) -> Result<()> {
    let span = meta.path.span();
    let value = meta.value()?.parse()?;
    set_once(slot, value, attr, span)
}

fn is_crate_attr(path: &Path) -> bool {
    path.segments.len() == 1
        && path
            .segments
            .first()
            .is_some_and(|segment| segment.ident == "crate")
}

fn default_ident_name(ident: &Ident) -> LitStr {
    let name = ident.to_string();
    let name = name.strip_prefix("r#").unwrap_or(&name);
    LitStr::new(name, ident.span())
}
