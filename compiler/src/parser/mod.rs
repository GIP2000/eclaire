#[cfg(test)]
mod test;

mod grammer;
pub mod symbol_table;
use std::marker::PhantomData;

use grammer::TranslationUnit;
use lexer::{ErrorMeta, LexerIterator, LexerIteratorError};

use crate::{
    lexer::{LexToken, MyLexerError},
    parser::symbol_table::SymbolTable,
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

#[inline]
pub fn safe_parse_wrapper<'a, I, O, F>(
    token_stream: &mut I,
    symbol_table: &mut SymbolTable,
    parser: &mut F,
) -> Result<O>
where
    I: LexerIterator<'a, LexToken<'a>, MyLexerError>,
    F: FnMut(&mut I, &mut SymbolTable) -> Result<O>,
{
    if let None | Some(Err(LexerIteratorError::NoMoreTokens)) = token_stream.clone().next() {
        return Err(LexerIteratorError::NoMoreTokens.into());
    };

    let mut copy = token_stream.clone();

    let result = parser(&mut copy, symbol_table).map_err(|err| match err {
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

trait Parse
where
    Self: Sized,
{
    fn from_lexer<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
        symbol_table: &mut SymbolTable,
    ) -> Result<Self>;

    fn from_lexer_safe<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
        symbol_table: &mut SymbolTable,
    ) -> Result<Self> {
        let result = safe_parse_wrapper(token_stream, symbol_table, &mut Self::from_lexer);

        if let Err(err) = &result {
            trace!("Error @ {err:?}");
        }

        result
    }

    fn from_lexer_many<'a, 'b, I: LexerIterator<'a, LexToken<'a>, MyLexerError>>(
        token_stream: &'b mut I,
        symbol_table: &'b mut SymbolTable,
    ) -> ParseIterWith<
        'a,
        'b,
        I,
        Self,
        fn(token_stream: &mut I, symbol_table: &mut SymbolTable) -> Result<Self>,
    > {
        ParseIterWith {
            iter: token_stream,
            func: Self::from_lexer_safe,
            symbol_table,
            phaton_data: PhantomData,
        }
    }
}

trait ParserInto<'a, Output: Parse>
where
    Self: LexerIterator<'a, LexToken<'a>, MyLexerError>,
{
    fn parse(&mut self, symbol_table: &mut SymbolTable) -> Result<Output> {
        Output::from_lexer_safe(self, symbol_table)
    }

    fn parse_many<'b>(
        &'b mut self,
        symbol_table: &'b mut SymbolTable,
    ) -> ParseIterWith<
        'a,
        'b,
        Self,
        Output,
        fn(token_stream: &mut Self, symbol_table: &mut SymbolTable) -> Result<Output>,
    > {
        Output::from_lexer_many(self, symbol_table)
    }
}

impl<'a, Output: Parse, I: LexerIterator<'a, LexToken<'a>, MyLexerError>> ParserInto<'a, Output>
    for I
{
}

impl<T: for<'c> TryFrom<&'c LexToken<'c>>> Parse for T {
    fn from_lexer<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
        _: &mut SymbolTable,
    ) -> Result<Self> {
        token_stream
            .next_matches_func(|x| T::try_from(x).ok())
            .map_err(|err| err.into())
    }
}

struct ParseIterWith<
    'a,
    'b,
    I: LexerIterator<'a, LexToken<'a>, MyLexerError>,
    O,
    F: FnMut(&mut I, &mut SymbolTable) -> Result<O>,
> {
    iter: &'b mut I,
    func: F,
    symbol_table: &'b mut SymbolTable,
    phaton_data: PhantomData<&'a O>,
}

impl<
        'a,
        'b,
        I: LexerIterator<'a, LexToken<'a>, MyLexerError>,
        O,
        F: FnMut(&mut I, &mut SymbolTable) -> Result<O>,
    > Iterator for ParseIterWith<'a, 'b, I, O, F>
{
    type Item = Result<O>;

    fn next(&mut self) -> Option<Self::Item> {
        match safe_parse_wrapper(self.iter, self.symbol_table, &mut self.func) {
            Err(ParserError::LexerError(LexerIteratorError::NoMoreTokens)) => None,
            x @ Ok(_) | x @ Err(_) => Some(x),
        }
    }
}

trait ParseIntoWith<'a>
where
    Self: LexerIterator<'a, LexToken<'a>, MyLexerError>,
{
    fn parse_with<Output>(
        &mut self,
        symbol_table: &mut SymbolTable,
        mut parser: impl FnMut(&mut Self, &mut SymbolTable) -> Result<Output>,
    ) -> Result<Output> {
        safe_parse_wrapper(self, symbol_table, &mut parser)
    }

    fn parse_with_many<'b, Output, F: FnMut(&mut Self, &mut SymbolTable) -> Result<Output>>(
        &'b mut self,
        symbol_table: &'b mut SymbolTable,
        parser: F,
    ) -> ParseIterWith<'a, 'b, Self, Output, F> {
        ParseIterWith {
            iter: self,
            symbol_table,
            func: parser,
            phaton_data: PhantomData,
        }
    }
}

impl<'a, I: LexerIterator<'a, LexToken<'a>, MyLexerError>> ParseIntoWith<'a> for I {}

pub fn parse(source_code: &str) -> Result<(TranslationUnit, SymbolTable)> {
    let mut lexer = LexToken::lex(source_code);
    trace!("starting parse");
    let mut symbol_table = SymbolTable::default();
    let result = lexer.parse(&mut symbol_table);
    trace!("finished parse");
    Ok((result?, symbol_table))
}
