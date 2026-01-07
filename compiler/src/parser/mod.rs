#[cfg(test)]
mod test;

mod grammer;
use std::marker::PhantomData;

use grammer::TranslationUnit;
use lexer::{LexerIterator, LexerIteratorError};

use crate::lexer::{LexToken, MyLexerError};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ParserError<E> {
    #[error("lexer error")]
    LexerError(#[from] LexerIteratorError<E>),
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
        let mut copy = token_stream.clone();
        let result = Self::from_lexer(&mut copy)?;
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
    lexer.parse()
}
