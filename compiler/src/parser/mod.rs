#[cfg(test)]
mod test;

mod grammer;
use std::marker::PhantomData;

use grammer::TranslationUnit;
use lexer::{ErrorMeta, LexerIterator, LexerIteratorError};

use crate::{
    lexer::{LexToken, MyLexerError},
    trace,
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ParserError<E> {
    #[error(transparent)]
    LexerError(#[from] LexerIteratorError<E>),
    #[error("Failed to find parsing target, not match found: {0}")]
    DoesNotMatch(ErrorMeta),
    #[error("Other Error")]
    Other,
}

pub type Result<T> = std::result::Result<T, ParserError<MyLexerError>>;

trait Parse
where
    Self: Sized,
{
    fn from_lexer<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
    ) -> Result<Self>;

    fn from_lexer_safe<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
    ) -> Result<Self> {
        // Only say there is no more tokens if it starts with no more tokens
        if let None | Some(Err(LexerIteratorError::NoMoreTokens)) = token_stream.clone().next() {
            return Err(LexerIteratorError::NoMoreTokens.into());
        };

        let mut copy = token_stream.clone();
        // Otherwise say that we Do not match if there is no more tokens
        let result = Self::from_lexer(&mut copy).map_err(|err| match err {
            ParserError::LexerError(lexer::LexerIteratorError::NoMoreTokens) => {
                // TODO: make this better with an actual lineno and colno
                ParserError::DoesNotMatch(ErrorMeta {
                    lineno: 0,
                    colno: 0,
                    display: "Unexpected EOF".into(),
                })
            }
            err => err,
        })?;

        *token_stream = copy;
        Ok(result)
    }

    fn from_lexer_many<'a, 'b, I: LexerIterator<'a, LexToken<'a>, MyLexerError>>(
        token_stream: &'b mut I,
    ) -> ParseIter<'a, 'b, I, Self> {
        ParseIter {
            iter: token_stream,
            phantom_data: PhantomData,
        }
    }
}

struct ParseIter<'a, 'b, I: LexerIterator<'a, LexToken<'a>, MyLexerError>, O: Parse> {
    iter: &'b mut I,
    phantom_data: PhantomData<&'a O>,
}

impl<'a, 'b, I: LexerIterator<'a, LexToken<'a>, MyLexerError>, T: Parse> std::iter::Iterator
    for ParseIter<'a, 'b, I, T>
{
    type Item = Result<T>;

    fn next(&mut self) -> Option<Self::Item> {
        match T::from_lexer_safe(self.iter) {
            Err(ParserError::LexerError(LexerIteratorError::NoMoreTokens)) => None,
            x @ Ok(_) | x @ Err(_) => Some(x),
        }
    }
}

trait ParserInto<'a, Output: Parse>
where
    Self: LexerIterator<'a, LexToken<'a>, MyLexerError>,
{
    fn parse(&mut self) -> Result<Output> {
        Output::from_lexer_safe(self)
    }

    fn parse_many<'b>(&'b mut self) -> ParseIter<'a, 'b, Self, Output> {
        Output::from_lexer_many(self)
    }
}

impl<'a, Output: Parse, I: LexerIterator<'a, LexToken<'a>, MyLexerError>> ParserInto<'a, Output>
    for I
{
}

pub fn parse(source_code: &str) -> Result<TranslationUnit> {
    let mut lexer = LexToken::lex(source_code);
    trace!("starting parse");
    let result = lexer.parse();
    trace!("finished parse");
    result
}
