use proc_macro2::Span;
use syn::{
    Attribute, Error, Field, Ident, LitStr, Path, Result, meta::ParseNestedMeta, parse_quote,
    spanned::Spanned,
};

pub(crate) struct TypeAttrs {
    name: LitStr,
    version: Option<LitStr>,
    schema_path: Path,
}

impl TypeAttrs {
    pub(crate) fn parse(attrs: &[Attribute], ident: &Ident) -> Result<Self> {
        let mut rename = None;
        let mut version = None;
        let mut schema_crate = None;
        for attr in attrs {
            if !attr.path().is_ident("cache_schema") {
                continue;
            }
            attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("rename") {
                    parse_lit_str_value(&mut rename, meta, "rename")
                } else if meta.path.is_ident("version") {
                    parse_lit_str_value(&mut version, meta, "version")
                } else if is_crate_attr(&meta.path) {
                    parse_path_value(&mut schema_crate, meta, "crate")
                } else {
                    Err(meta.error("unsupported cache_schema type attribute"))
                }
            })?;
        }
        Ok(Self {
            name: rename.unwrap_or_else(|| default_ident_name(ident)),
            version,
            schema_path: schema_crate.unwrap_or_else(|| parse_quote!(::ssa_cache_schema)),
        })
    }

    pub(crate) fn schema_path(&self) -> &Path {
        &self.schema_path
    }

    pub(crate) fn name(&self) -> &LitStr {
        &self.name
    }

    pub(crate) fn version(&self) -> Option<&LitStr> {
        self.version.as_ref()
    }
}

pub(crate) struct VariantAttrs {
    name: LitStr,
}

impl VariantAttrs {
    pub(crate) fn parse(attrs: &[Attribute], ident: &Ident) -> Result<Self> {
        let mut rename = None;
        for attr in attrs {
            if !attr.path().is_ident("cache_schema") {
                continue;
            }
            attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("rename") {
                    parse_lit_str_value(&mut rename, meta, "rename")
                } else {
                    Err(meta.error("unsupported cache_schema variant attribute"))
                }
            })?;
        }
        Ok(Self {
            name: rename.unwrap_or_else(|| default_ident_name(ident)),
        })
    }

    pub(crate) fn name(&self) -> &LitStr {
        &self.name
    }
}

#[derive(Default)]
pub(crate) struct FieldAttrs {
    name: Option<LitStr>,
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
                    parse_lit_str_value(&mut parsed.name, meta, "rename")
                } else {
                    Err(meta.error("unsupported cache_schema field attribute"))
                }
            })?;
        }
        if parsed.name.is_none()
            && let Some(ident) = &field.ident
        {
            parsed.name = Some(default_ident_name(ident));
        }
        Ok(parsed)
    }

    pub(crate) fn name(&self) -> Option<&LitStr> {
        self.name.as_ref()
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
