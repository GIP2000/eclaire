use lexer::LexerIteratorError;
use thiserror::Error;

use crate::{
    lexer::MyLexerError,
    parser::{
        Parser,
        grammer::statment::{IdentCreation, IdentCreationType},
    },
    utils::iterator::IterPlusError,
};

pub mod expression;
pub mod function;
pub mod ident;
pub mod statment;
pub mod structure;
pub mod types;

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    LexerIteratorError(LexerIteratorError<MyLexerError>),

    #[error("Does not match: {0}")]
    DoesNotMatch(&'static str),
}

impl From<LexerIteratorError<MyLexerError>> for Error {
    fn from(value: LexerIteratorError<MyLexerError>) -> Self {
        Self::LexerIteratorError(value)
    }
}

pub type Result<T> = core::result::Result<T, Error>;

#[derive(Debug)]
pub struct TranslationUnion<'a>(pub Box<[IdentCreation<'a>]>);

impl<'a> Parser<'a> for TranslationUnion<'a> {
    type Error = Error;

    fn from_lexer<L: crate::lexer::MyLexer<'a>>(
        lexer: &mut L,
    ) -> std::result::Result<Self, Self::Error> {
        let IterPlusError(v, err) = IdentCreation::parse_many(lexer)
            .map(|x| match x {
                r @ Ok(IdentCreation {
                    ic_type: IdentCreationType::Const,
                    ident: _,
                    type_val: _,
                    lvalue: _,
                }) => r,
                Ok(_) => Err(Error::DoesNotMatch(
                    "Ident Cration pattern must be Const Type",
                )),
                Err(err) => Err(err),
            })
            .collect();

        match err {
            Some(Error::LexerIteratorError(LexerIteratorError::NoMoreTokens)) => Ok(Self(v)),
            Some(err) => Err(err),
            None => Err(Error::DoesNotMatch("Empty File")),
        }
    }
}
