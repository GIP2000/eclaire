use std::marker::PhantomData;

use crate::{
    lexer::LexToken,
    parser::{
        Parser, ParserWithState as _,
        grammer::{ident::Ident, types::Type},
    },
};

#[derive(Debug, PartialEq)]
pub struct Fields<'a>(Box<[(Ident<'a>, Type<'a>)]>);

impl<'a> Parser<'a> for Fields<'a> {
    type Error = super::Error;

    fn from_lexer<L: crate::lexer::MyLexer<'a>>(lexer: &mut L) -> Result<Self, Self::Error> {
        _ = lexer.next_matches(LexToken::OCBracket)?;

        let mut must_stop = false;

        let fields: Box<_> = (|lexer: &mut L| {
            if must_stop {
                anyhow::bail!("");
            }
            let ident = Ident::parse(lexer)?;
            _ = lexer.next_matches(LexToken::Colon)?;
            let typ = Type::parse(lexer)?;
            if let Err(_) = lexer.next_matches(LexToken::Comma) {
                must_stop = true;
            }

            Ok((ident, typ))
        })
        .parse_many(lexer)
        .map_while(Result::ok)
        .collect();

        if !must_stop && fields.len() > 1 {
            return Err(super::Error::DoesNotMatch("Invalid Field"));
        }

        _ = lexer.next_matches(LexToken::CCBracket)?;

        Ok(Self(fields))
    }
}

#[derive(Debug, PartialEq)]
pub struct Struct<'a>(Fields<'a>);

impl<'a> Parser<'a> for Struct<'a> {
    type Error = super::Error;

    fn from_lexer<L: crate::lexer::MyLexer<'a>>(lexer: &mut L) -> Result<Self, Self::Error> {
        lexer
            .next_matches(LexToken::Struct)
            .map_err(Into::into)
            .and_then(|_| Fields::parse(lexer).map(|x| Self(x)))
    }
}

#[derive(Debug, PartialEq)]
pub struct Union<'a>(Fields<'a>);

impl<'a> Parser<'a> for Union<'a> {
    type Error = super::Error;

    fn from_lexer<L: crate::lexer::MyLexer<'a>>(lexer: &mut L) -> Result<Self, Self::Error> {
        lexer
            .next_matches(LexToken::Union)
            .map_err(Into::into)
            .and_then(|_| Fields::parse(lexer).map(|x| Self(x)))
    }
}

// TODO: implement this
#[derive(Debug, PartialEq)]
pub struct Enum<'a>(PhantomData<&'a ()>);

impl<'a> Parser<'a> for Enum<'a> {
    type Error = super::Error;

    fn from_lexer<L: crate::lexer::MyLexer<'a>>(_: &mut L) -> Result<Self, Self::Error> {
        Err(super::Error::DoesNotMatch("Enum Not yet implemented"))
        // TODO: Actually implement
    }
}
