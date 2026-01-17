pub mod expression;
pub mod ident;
pub mod statment;
pub mod structures;
use std::collections::HashMap;

use lexer::LexerIterator;

use super::{Parse, Result};
use crate::{
    lexer::{LexToken, MyLexerError},
    parser::{
        grammer::{
            ident::{Ident, IdentPair},
            statment::Statment,
        },
        symbol_table::{self, SymbolTable, TypeInfo},
        ParserInto,
    },
    trace,
    utils::iterator::IterPlusError,
};

#[derive(Debug)]
pub struct TranslationUnit {
    pub symbol_table: SymbolTable, // pub functions: Vec<Function>,
}

impl Parse for TranslationUnit {
    fn from_lexer<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
    ) -> Result<Self> {
        trace!("Entering TranslationUnit");

        let functions: HashMap<Ident, TypeInfo> = token_stream
            .parse_many()
            .map(|x: Result<Function>| x.map(|x| (x.name.clone(), x.into())))
            .collect::<Result<_>>()?;

        let symbol_table = SymbolTable {
            type_defs: functions,
            decls: HashMap::new(),
        };

        Ok(Self { symbol_table })
    }
}

#[derive(Debug)]
pub struct Function {
    pub name: Ident,
    pub args: Vec<IdentPair>,
    pub ret: Option<Ident>,
    pub statments: Vec<Statment>,
}

impl Parse for Function {
    fn from_lexer<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
    ) -> Result<Self> {
        trace!("entering Function");
        token_stream.next_matches(LexToken::Fn)?;

        let name: Ident = token_stream.parse()?;

        token_stream.next_matches(LexToken::OParen)?;

        let args = token_stream
            .parse_many()
            .take_while(|x| x.is_ok())
            .collect::<Result<_>>()
            .expect("Unreachable: only valid entries of ident pair");

        token_stream.next_matches(LexToken::CParen)?;

        let ret: Option<Ident> = token_stream
            .next_matches(LexToken::SkinnyArrow)
            .ok()
            .map(|_| token_stream.parse())
            .transpose()?;

        token_stream.next_matches(LexToken::OCBracket)?;

        let parse_iter = token_stream.parse_many();

        let IterPlusError(statments, following) = parse_iter.collect();

        token_stream
            .next_matches(LexToken::CCBracket)
            .map_err(|err| following.unwrap_or(err.into()))?;

        Ok(Self {
            name,
            args,
            ret,
            statments,
        })
    }
}
