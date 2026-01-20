pub mod assignment;
pub mod expression;
pub mod ident;
pub mod statment;
pub mod structures;

use lexer::LexerIterator;

use super::{Parse, Result};
use crate::{
    lexer::{LexToken, MyLexerError},
    parser::{
        grammer::{
            assignment::{Assignment, AssignmentType},
            ident::{Ident, IdentPair},
            statment::Statment,
        },
        symbol_table::SymbolTable,
        ParserInto,
    },
    trace,
    utils::iterator::IterPlusError,
};

#[derive(Debug)]
pub struct TranslationUnit(pub Vec<Assignment>);

impl Parse for TranslationUnit {
    fn from_lexer<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
        symbol_table: &mut SymbolTable,
    ) -> Result<Self> {
        let IterPlusError(result, following): IterPlusError<Vec<_>> = token_stream
            .parse_many(symbol_table)
            .map(|x| match x {
                x @ Err(_)
                | x @ Ok(Assignment {
                    assignment_type: AssignmentType::Const,
                    ident: _,
                    data_type: _,
                    expr: _,
                }) => x,
                _ => Err(super::ParserError::Other),
            })
            .collect();

        if let Some(following) = following {
            return Err(following);
        }

        Ok(Self(result)) // Is this neccisary
    }
}

#[derive(Debug, Clone)]
pub struct Function {
    pub args: Vec<IdentPair>,
    pub ret: Option<Ident>,
    pub statments: Vec<Statment>,
    pub symbol_table_idx: usize,
}

impl Parse for Function {
    fn from_lexer<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
        symbol_table: &mut SymbolTable,
    ) -> Result<Self> {
        trace!("entering Function");

        token_stream.next_matches(LexToken::Fn)?;

        token_stream.next_matches(LexToken::OParen)?;

        let args = token_stream
            .parse_many(symbol_table)
            .take_while(|x| x.is_ok())
            .collect::<Result<_>>()
            .expect("Unreachable: only valid entries of ident pair");

        token_stream.next_matches(LexToken::CParen)?;

        let ret: Option<Ident> = token_stream
            .next_matches(LexToken::SkinnyArrow)
            .ok()
            .map(|_| token_stream.parse(symbol_table))
            .transpose()?;

        token_stream.next_matches(LexToken::OCBracket)?;

        let symbol_table_idx = symbol_table.push();

        let IterPlusError(statments, following) = token_stream.parse_many(symbol_table).collect();

        symbol_table
            .pop()
            .expect("I just pushed this should never happen");

        token_stream
            .next_matches(LexToken::CCBracket)
            .map_err(|err| following.unwrap_or(err.into()))?;

        Ok(Self {
            args,
            ret,
            statments,
            symbol_table_idx,
        })
    }
}
