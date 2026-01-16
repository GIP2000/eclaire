pub mod expression;
pub mod ident;
use lexer::LexerIterator;

use super::{Parse, Result};
use crate::{
    lexer::{LexToken, MyLexerError},
    parser::{
        grammer::{
            expression::Expression,
            ident::{Ident, IdentPair},
        },
        ParserInto,
    },
    trace,
    utils::iterator::IterPlusError,
};

#[derive(Debug)]
pub struct TranslationUnit {
    pub functions: Vec<Function>,
}

impl Parse for TranslationUnit {
    fn from_lexer<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
    ) -> Result<Self> {
        trace!("Entering TranslationUnit");

        Ok(Self {
            functions: token_stream.parse_many().collect::<Result<_>>()?,
        })
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

#[derive(Debug)]
pub enum Statment {
    Assignment(Ident, Option<Ident>, Expression), // do I need the option? can I figure out the
    // datatype always and replace this with an ident
    // pair?
    Expression(Expression),
    // add loops
    // add match
}

impl Parse for Statment {
    fn from_lexer<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
    ) -> Result<Self> {
        trace!("Entering Statment");
        let mut temp_token_stream = token_stream.clone();

        let ident = temp_token_stream
            .next_matches(LexToken::Let)
            .ok()
            .map(|_| -> Result<_> {
                trace!("Entering Let Statment");
                let result = temp_token_stream.parse()?;
                temp_token_stream.next_matches(LexToken::Eq)?;
                *token_stream = temp_token_stream;
                Ok(result)
            })
            .transpose()?;

        let expression: Expression = token_stream.parse()?;
        token_stream.next_matches(LexToken::SemiColon)?;

        match ident {
            Some((ident, datatype)) => return Ok(Self::Assignment(ident, datatype, expression)),
            None => Ok(Self::Expression(expression)),
        }
    }
}
