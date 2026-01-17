use lexer::LexerIterator;

use crate::{
    lexer::{LexToken, MyLexerError},
    parser::{
        grammer::{expression::Expression, ident::Ident, Function},
        symbol_table::SymbolTable,
        Parse, ParserInto, Result,
    },
    trace,
};

#[derive(Debug)]
pub enum Statment {
    Assignment(Ident, Option<Ident>, Expression),
    Expression(Expression),
}

impl Parse for Statment {
    fn from_lexer<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
        symbol_table: &mut SymbolTable,
    ) -> Result<Self> {
        trace!("Entering Statment");
        let mut temp_token_stream = token_stream.clone();

        let ident = temp_token_stream
            .next_matches(LexToken::Let)
            .ok()
            .map(|_| -> Result<_> {
                trace!("Entering Let Statment");
                let result = temp_token_stream.parse(symbol_table)?;
                temp_token_stream.next_matches(LexToken::Eq)?;
                *token_stream = temp_token_stream;
                Ok(result)
            })
            .transpose()?;

        let expression: Expression = token_stream.parse(symbol_table)?;
        token_stream.next_matches(LexToken::SemiColon)?;

        match ident {
            Some((ident, datatype)) => return Ok(Self::Assignment(ident, datatype, expression)),
            None => Ok(Self::Expression(expression)),
        }
    }
}
