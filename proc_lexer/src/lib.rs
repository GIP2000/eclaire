extern crate proc_macro;

use lexer::{
    dfa::{DFABoxed, DFA, DFA_SIZE},
    AcceptFunc,
};
use proc_macro::TokenStream;

use quote::quote;
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input, Data, DeriveInput, Ident, LitStr,
};

struct RegexAttributeArgs {
    regex_pattern: Box<str>,
    func_name: Option<BoxStr>,
}

#[derive(Clone, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
struct BoxStr(Box<str>);

impl AcceptFunc for BoxStr {
    type Output<'a> = &'a str;

    fn convert<'a>(&self, input: &'a str) -> anyhow::Result<Self::Output<'a>> {
        Ok(input)
    }
}

impl Parse for RegexAttributeArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let regex_pattern: LitStr = input.parse()?;
        let regex_pattern = regex_pattern.value().into();

        eprintln!("regex_pattern = {}", regex_pattern);

        let _comma: syn::Result<syn::token::Comma> = input.parse();

        let func_name = match input.parse::<Ident>() {
            Ok(func_ident) => {
                if func_ident != "func" {
                    return Err(input.error("Expected `func` as the second argument key"));
                }

                let _eq_token: syn::Token![=] = input.parse()?;

                let func_name: Ident = input.parse()?;

                Some(BoxStr(func_name.to_string().into()))
            }
            Err(_) => None,
        };

        Ok(RegexAttributeArgs {
            regex_pattern,
            func_name,
        })
    }
}

#[proc_macro_derive(Lexer, attributes(regex))]
pub fn build_dfa(input: TokenStream) -> TokenStream {
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

                    return Some(((regex, BoxStr(name.to_string().into())), Some(func_impl)));
                }
                _ => {
                    panic!("#[build_dfa] func required if variant has data");
                }
            };
        })
    });
    let (regexes, funcs): (Vec<_>, Vec<_>) = regexes.unzip();

    let dfa: DFABoxed<BoxStr> = match DFABoxed::from_regexes(regexes.into_iter()) {
        Ok(x) => x,
        Err(e) => {
            panic!("Failed to compile regexes to dfa: {e:?}");
        }
    };

    let funcs: Box<_> = funcs.into_iter().filter_map(|x| x).collect();

    let state_count = dfa.states_len();

    let d_trans: Box<_> = dfa
        .d_trans
        .into_iter()
        .map(|state| {
            let inner: Box<_> = state
                .into_iter()
                .map(|trans| {
                    use lexer::dfa::TransitionType::*;

                    fn make_ident(f: &str) -> syn::Ident {
                        syn::Ident::new(f, proc_macro2::Span::call_site())
                    }

                    let result = match trans {
                        Normal(x) => quote! {lexer::dfa::TransitionType::Normal(#x)},
                        Fail => quote! {lexer::dfa::TransitionType::make_fail()},
                        Accpet(f) => {
                            let f = make_ident(&f.0.trim());
                            quote! {lexer::dfa::TransitionType::Accpet(FnPContainer(#f))}
                        }
                        AccpetOr(x, f) => {
                            let f = make_ident(&f.0.trim());
                            quote! {lexer::dfa::TransitionType::AccpetOr(#x, FnPContainer(#f))}
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

    let enum_name_for_impl = if *lifetime_count == 1 {
        quote! {#enum_name<'a>}
    } else {
        quote! {#enum_name}
    };

    let result = quote! {


        impl<'a> lexer::Lexer<'a, __lexer_gen__::FnPContainer, lexer::dfa::DFAStatic<#state_count, #DFA_SIZE, __lexer_gen__::FnPContainer>> for #enum_name_for_impl {

            fn lex<'d>(
                input: &'a str,
            ) -> lexer::Lex<'a, 'd, __lexer_gen__::FnPContainer, lexer::dfa::DFAStatic<#state_count, #DFA_SIZE, __lexer_gen__::FnPContainer>> {
                use lexer::dfa::DFA;
                __lexer_gen__::LexTokenDFA.lex(input)
            }
        }
        mod __lexer_gen__ {
            use super::*;

            pub(crate) type DFAType = lexer::dfa::DFAStatic<#state_count, #DFA_SIZE, FnPContainer>;

            pub(crate) type LexerType<'a> = lexer::Lex<'a, 'static, __lexer_gen__::FnPContainer, DFAType>;


            #[derive(Clone)]
            pub struct FnPContainer(for<'a> fn(&'a str) -> anyhow::Result<#enum_name_for_impl>);

            impl lexer::AcceptFunc for FnPContainer {
                type Output<'a> = #enum_name_for_impl;

                fn convert<'a>(&self, input: &'a str) -> anyhow::Result<Self::Output<'a>> {
                    self.0(input)
                }
            }


            pub static #dfa_name: DFAType = lexer::dfa::DFAStatic {
                d_trans: #arr,
            };

            #(#funcs)*
        }

    };

    result.into()
}
