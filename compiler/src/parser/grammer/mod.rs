pub mod expression;
use lexer::LexerIterator;

use super::{Parse, Result};
use crate::{
    lexer::{LexToken, MyLexerError},
    parser::{grammer::expression::Expression, ParserInto},
    trace,
};

#[derive(Debug)]
pub struct TranslationUnit {
    pub functions: Vec<Function>,
}

impl Parse for TranslationUnit {
    fn from_lexer<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
    ) -> Result<Self> {
        trace!("entering TranslationUnit");
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
            .take_while(|x| matches!(x, Ok(_)))
            .collect::<Result<_>>()
            .expect("Unreachable: only valid entries of ident pair");
        token_stream.next_matches(LexToken::CParen)?;

        let ret: Option<Ident> = if let Ok(_) = token_stream.next_matches(LexToken::SkinnyArrow) {
            Some(token_stream.parse()?)
        } else {
            None
        };

        token_stream.next_matches(LexToken::OCBracket)?;

        let statments: Vec<Statment> = token_stream
            .parse_many()
            .take_while(Result::is_ok)
            .map(Result::unwrap)
            .collect();

        token_stream.next_matches(LexToken::CCBracket)?;

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

impl Parse for (Ident, Option<Ident>) {
    fn from_lexer<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
    ) -> Result<Self> {
        token_stream
            .parse()
            .map(|x: IdentPair| (x.name, Some(x.datatype)))
            .or_else(|_| token_stream.parse().map(|x: Ident| (x, None)))
    }
}

impl Parse for Statment {
    fn from_lexer<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
    ) -> Result<Self> {
        trace!("Entering Statment");
        let mut temp_token_stream = token_stream.clone();
        let ident: Result<(Ident, Option<Ident>)> = temp_token_stream
            .next_matches(LexToken::Let)
            .map_err(|err| err.into())
            .and_then(|_| temp_token_stream.parse())
            .or_else(|_| temp_token_stream.parse())
            .and_then(|x| {
                temp_token_stream.next_matches(LexToken::Eq)?;
                *token_stream = temp_token_stream;
                Ok(x)
            });

        let expression: Expression = token_stream.parse()?;
        token_stream.next_matches(LexToken::SemiColon)?;

        match ident {
            Ok((ident, datatype)) => return Ok(Self::Assignment(ident, datatype, expression)),
            Err(_) => Ok(Self::Expression(expression)),
        }
    }
}

#[derive(Debug)]
pub struct Ident {
    pub value: Box<str>,
}

impl<A> PartialEq<A> for Ident
where
    A: AsRef<str>,
{
    fn eq(&self, other: &A) -> bool {
        &*self.value == other.as_ref()
    }
}

impl Parse for Ident {
    fn from_lexer<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
    ) -> Result<Self> {
        trace!("Entering Ident");
        token_stream
            .next_matches_func(|x| {
                if let LexToken::Ident(x) = x {
                    Some(x)
                } else {
                    None
                }
            })
            .map(|x| Ident { value: x.into() })
            .map_err(|x| x.into())
    }
}

#[derive(Debug)]
pub struct IdentPair {
    pub name: Ident,
    pub datatype: Ident,
}

impl Parse for IdentPair {
    fn from_lexer<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
    ) -> Result<Self> {
        trace!("Entering Ident Pair");
        let name = token_stream.parse()?;
        token_stream.next_matches(LexToken::Colon)?;
        let datatype = token_stream.parse()?;
        _ = token_stream.next_matches(LexToken::Comma);
        Ok(Self { name, datatype })
    }
}
