use lexer::LexerIterator;

use crate::{
    lexer::{LexToken, MyLexerError},
    parser::{
        grammer::{assignment::Assignment, expression::Expression},
        symbol_table::SymbolTableType,
        Parse, ParserInto, Result,
    },
    trace,
};

#[derive(Debug, Clone)]
pub enum Statment {
    Assignment(Assignment),
    Expression(Expression),
}

impl Parse for Statment {
    fn from_lexer<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
        symbol_table: &mut SymbolTableType,
    ) -> Result<Self> {
        trace!("Entering Statment");

        token_stream
            .parse(symbol_table)
            .map(|x| Self::Assignment(x))
            .or_else(|_| {
                token_stream
                    .parse(symbol_table)
                    .map(|x| Self::Expression(x))
                    .and_then(|x| {
                        token_stream
                            .next_matches(LexToken::SemiColon)
                            .map_err(|err| err.into())
                            .map(|_| x)
                    })
            })
    }
}
