extern crate proc_macro;
mod dfa;
mod trie;

use lexer::DFA;
use proc_macro::TokenStream;

use quote::quote;
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input, Data, DeriveInput, Ident, LitStr,
};

use dfa::DFABoxed;

use crate::dfa::DFA_SIZE;

struct RegexAttributeArgs {
    regex_pattern: Box<str>,
    func_name: Box<str>,
}

impl Parse for RegexAttributeArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        eprintln!("starting");
        // Parse the first argument, which should be the regex string literal
        let regex_pattern: LitStr = input.parse()?;
        eprintln!("litstr = {:?}", regex_pattern);
        let regex_pattern = regex_pattern.value().into();
        // let regex_pattern = regex_pattern.token().to_string().into();
        eprintln!("pattern = {}", regex_pattern);

        // Expect a comma after the regex pattern
        let _comma_token: syn::token::Comma = input.parse()?;

        // Parse the identifier for 'func'
        let func_ident: Ident = input.parse()?;
        if func_ident != "func" {
            return Err(input.error("Expected `func` as the second argument key"));
        }

        // Parse the `=` sign
        let _eq_token: syn::Token![=] = input.parse()?;

        // Parse the identifier for the function name
        let func_name: Ident = input.parse()?;

        let func_name = func_name.to_string().into();

        eprintln!("finishing {:?} {:?}", regex_pattern, func_name);

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

    let regexes = data.variants.iter().flat_map(|variant| {
        variant.attrs.iter().filter_map(|att| {
            if !att.path().is_ident("regex") {
                return None;
            }
            att.parse_args::<RegexAttributeArgs>()
                .ok()
                .map(|x| (x.regex_pattern, x.func_name))
        })
    });

    let dfa = match DFABoxed::from_regexes(regexes) {
        Ok(x) => x,
        Err(e) => {
            panic!("Failed to compile regexes to dfa: {e:?}");
        }
    };

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


    };

    result.into()
}
