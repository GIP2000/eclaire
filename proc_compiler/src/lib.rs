extern crate proc_macro;

use proc_macro::TokenStream;

use quote::quote;
use syn::{Data, DeriveInput, parse_macro_input};

#[proc_macro_derive(
    FromLexValue,
    attributes(source, left, skip, generic_impl_override, generic_source_override)
)]
pub fn from_token(input: TokenStream) -> TokenStream {
    let mut input_enum = parse_macro_input!(input as DeriveInput);

    let source_type = input_enum
        .attrs
        .iter()
        .find_map(|att| {
            att.path()
                .is_ident("source")
                .then_some(())
                .and_then(|_| att.parse_args::<syn::TypePath>().ok())
                .map(|x| quote! {#x})
        })
        .expect("#[from_token] must have a source");

    let data = match &mut input_enum.data {
        Data::Enum(data_enum) => data_enum,
        _ => panic!("#[from_token] can only be applied to enums"),
    };

    let ident = &input_enum.ident;

    let generics = &input_enum.generics;

    let generic_impl_override = input_enum
        .attrs
        .iter()
        .find_map(|att| {
            att.path()
                .is_ident("generic_impl_override")
                .then_some(())
                .and_then(|_| att.parse_args::<syn::Generics>().ok())
        })
        .unwrap_or(generics.clone());

    let generic_source_override = input_enum
        .attrs
        .iter()
        .find_map(|att| {
            att.path()
                .is_ident("generic_source_override")
                .then_some(())
                .and_then(|_| att.parse_args::<syn::Generics>().ok())
        })
        .unwrap_or(generics.clone());

    let (v1, v2): (Vec<_>, Vec<_>) = data
        .variants
        .iter()
        .filter_map(|x| {
            let right_name = &x.ident;

            let left_name = x
                .attrs
                .iter()
                .find_map(|att| {
                    att.path().is_ident("left").then_some(()).map(|_| {
                        att.parse_args::<syn::Ident>()
                            .expect("Must be a valid type path")
                    })
                })
                .unwrap_or(right_name.clone());

            let skip = x.attrs.iter().find(|att| att.path().is_ident("skip"));
            if let Some(_) = skip {
                return None;
            }

            Some(match x.fields {
                syn::Fields::Unnamed(_) | syn::Fields::Named(_) => (
                    quote! {
                            #source_type::#left_name(x) => Ok(Self::#right_name(x))
                    },
                    quote! {
                            #ident::#right_name(x) => Self::#left_name(x)
                    },
                ),
                syn::Fields::Unit => (
                    quote! {
                            #source_type::#left_name => Ok(Self::#right_name)
                    },
                    quote! {
                            #ident::#right_name => Self::#left_name
                    },
                ),
            })
        })
        .collect();

    let from = (v2.len() == data.variants.len()).then_some(quote! {

        impl #generic_impl_override std::convert::From<#ident #generics> for #source_type #generic_source_override {
            fn from(value: #ident #generics) -> Self {
                match value {
                    #(#v2),*
                }
            }
        }
    });

    quote! {
        impl #generic_impl_override std::convert::TryFrom<#source_type #generic_source_override> for #ident #generics {
            type Error = ();

            fn try_from(value: #source_type #generic_source_override) -> core::result::Result<Self, Self::Error> {
                match value {
                    #(#v1,)*
                    _ => Err(()),
                }
            }
        }

        #from

    }
    .into()
}
