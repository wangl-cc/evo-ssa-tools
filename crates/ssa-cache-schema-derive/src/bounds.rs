use proc_macro2::TokenStream as TokenStream2;
use syn::{Data, Generics, Type, parse_quote};

pub(crate) fn add_field_schema_bounds(
    generics: &mut Generics,
    data: &Data,
    schema_path: &TokenStream2,
) {
    match data {
        Data::Struct(data) => add_bounds(
            generics,
            data.fields.iter().map(|field| &field.ty),
            schema_path,
        ),
        Data::Enum(data) => add_bounds(
            generics,
            data.variants
                .iter()
                .flat_map(|variant| variant.fields.iter().map(|field| &field.ty)),
            schema_path,
        ),
        Data::Union(_) => {}
    }
}

fn add_bounds<'a, I>(generics: &mut Generics, field_types: I, schema_path: &TokenStream2)
where
    I: IntoIterator<Item = &'a Type>,
{
    let mut field_types = field_types.into_iter().peekable();
    if field_types.peek().is_none() {
        return;
    }

    let where_clause = generics.make_where_clause();
    for ty in field_types {
        where_clause
            .predicates
            .push(parse_quote!(#ty: #schema_path::CacheSchema));
    }
}
