use lexer::LexerIterator;

use super::{Parse, Result};
use crate::{
    lexer::{LexToken, MyLexerError},
    parser::{ParserError, ParserInto},
};

#[derive(Debug)]
pub struct TranslationUnit {
    pub functions: Vec<Function>,
}

impl Parse for TranslationUnit {
    fn from_lexer<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
    ) -> Result<Self> {
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
}

impl Parse for Function {
    fn from_lexer<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
    ) -> Result<Self> {
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

        //TODO: stuff in the middle

        token_stream.next_matches(LexToken::CCBracket)?;

        Ok(Self { name, args, ret })
    }
}

pub enum Statment {
    Expression(Expression),
}

pub enum Expression {
    Ident(Ident),
    BinaryOp(Box<Expression>, BinaryOperators, Box<Expression>),
}

pub enum BinaryOperators {
    Eq,
}

impl<'a> TryFrom<LexToken<'a>> for BinaryOperators {
    type Error = ParserError<MyLexerError>;

    fn try_from(value: LexToken<'a>) -> Result<Self> {
        match value {
            LexToken::Eq => Ok(Self::Eq),
            _ => Err(lexer::LexerIteratorError::DoesNotMatch.into()),
        }
    }
}

impl Parse for BinaryOperators {
    fn from_lexer<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
    ) -> Result<Self> {
        // let op: Option<Self> = token_stream.next_matches_func(|x| {
        //     let x: Result<BinaryOperators> = x.try_into();
        //     x
        // });
        todo!()
        // op
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
        let name = token_stream.parse()?;
        token_stream.next_matches(LexToken::Colon)?;
        let datatype = token_stream.parse()?;
        _ = token_stream.next_matches(LexToken::Comma);
        Ok(Self { name, datatype })
    }
}
