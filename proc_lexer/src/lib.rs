extern crate proc_macro;
mod dfa;
mod trie;

use dfa::{DFABoxed, DFA_SIZE};
use lexer::DFA;
use proc_macro::TokenStream;

use quote::quote;
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input, Data, DeriveInput, Ident, LitStr,
};

struct RegexAttributeArgs {
    regex_pattern: Box<str>,
    func_name: Option<Box<str>>,
}

impl Parse for RegexAttributeArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let regex_pattern: LitStr = input.parse()?;
        let regex_pattern = regex_pattern.value().into();

        let _comma: syn::Result<syn::token::Comma> = input.parse();

        let func_name = match input.parse::<Ident>() {
            Ok(func_ident) => {
                if func_ident != "func" {
                    return Err(input.error("Expected `func` as the second argument key"));
                }

                let _eq_token: syn::Token![=] = input.parse()?;

                let func_name: Ident = input.parse()?;

                Some(func_name.to_string().into())
            }
            Err(_) => None,
        };

        Ok(RegexAttributeArgs {
            regex_pattern,
            func_name,
        })
    }
}

#[proc_macro_attribute]
pub fn build_dfa(_att: TokenStream, input: TokenStream) -> TokenStream {
    let mut input_enum = parse_macro_input!(input as DeriveInput);
    let enum_name = &input_enum.ident;

    let data = match &mut input_enum.data {
        Data::Enum(data_enum) => data_enum,
        _ => panic!("#[build_dfa] can only be applied to enums"),
    };

    let lifetime_count = &input_enum.generics.lifetimes().count();

    let regexes = data.variants.iter().flat_map(|variant| {
        let ident = &variant.ident;
        variant.attrs.iter().enumerate().filter_map(move |(ia, att)| {
            if !att.path().is_ident("regex") {
                return None;
            }
            let (regex, maybe_name) = att
                .parse_args::<RegexAttributeArgs>()
                .ok()
                .map(|x| (x.regex_pattern, x.func_name))?;

            match (maybe_name, &variant.fields) {
                (Some(name), _) => return Some(((regex, name), None)),
                (None, syn::Fields::Unit) => {
                    let name = Ident::new(
                        &format!("__parse_{}_{}__", ident, ia),
                        proc_macro2::Span::call_site(),
                    );

                    let func_impl = match lifetime_count {
                        1 => {
                            quote! {fn #name<'a>(input: &'a str) -> anyhow::Result<#enum_name<'a>> {
                                Ok(#enum_name::#ident)
                            }}
                        }
                        0 => quote! {fn #name<'a>(input: &'a str) -> anyhow::Result<#enum_name> {
                            Ok(#enum_name::#ident)
                        }},
                        _ => panic!("Invalid amount of lifetime parameters"),
                    };
                    return Some(((regex, name.to_string().into()), Some(func_impl)));
                }
                _ => {
                    panic!("#[build_dfa] func required if variant has data");
                }
            };
        })
    });
    let (regexes, funcs): (Vec<_>, Vec<_>) = regexes.unzip();

    let dfa = match DFABoxed::from_regexes(regexes.into_iter()) {
        Ok(x) => x,
        Err(e) => {
            panic!("Failed to compile regexes to dfa: {e:?}");
        }
    };

    let funcs: Box<_> = funcs.into_iter().filter_map(|x| x).collect();

    data.variants.iter_mut().for_each(|variant| {
        variant.attrs.retain(|attr| !attr.path().is_ident("regex"));
    });

    let state_count = dfa.states_len();

    let d_trans: Box<_> = dfa
        .d_trans
        .into_iter()
        .map(|state| {
            let inner: Box<_> = state
                .into_iter()
                .map(|trans| {
                    use dfa::TransitionType::*;

                    let make_ident = |f: &str| syn::Ident::new(f, proc_macro2::Span::call_site());

                    let result = match trans {
                        Normal(x) => quote! {lexer::TransitionType::Normal(#x)},
                        Fail => quote! {lexer::TransitionType::Fail},
                        Accpet(f) => {
                            let f = make_ident(&f.trim());
                            quote! {lexer::TransitionType::Accpet(#f)}
                        }
                        AccpetOr(x, f) => {
                            let f = make_ident(&f.trim());
                            quote! {lexer::TransitionType::AccpetOr(#x, #f)}
                        }
                    };

                    result
                })
                .collect();

            quote! {[ #(#inner),* ]}
        })
        .collect();

    let arr = quote! {
        [ #(#d_trans),* ]
    };

    let dfa_name = syn::Ident::new(&format!("{}DFA", enum_name), proc_macro2::Span::call_site());

    let result = quote! {
        #input_enum

        static #dfa_name: lexer::DFAStatic<#state_count, #DFA_SIZE, #enum_name> = lexer::DFAStatic {
            d_trans: #arr,
        };

        #(#funcs)*

    };

    result.into()
}
