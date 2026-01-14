use lexer::LexerIterator;

use crate::{
    lexer::{LexToken, MyLexerError},
    parser::{Parse, ParserError, ParserInto, Result},
    trace,
};

use super::Ident;

#[derive(Debug)]
pub enum Expression {
    // TODO: Figure out how to do this nested shit
    // BinaryOp(Box<Expression>, BinaryOperator, Box<Expression>),
    // UnaryOp(Box<Expression>, UnaryOperator),
    Ident(Ident),
    Constant(Box<str>), // TODO: enrich this type / expand it with more stuff
}

impl Parse for Expression {
    fn from_lexer<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
    ) -> Result<Self> {
        trace!("Entering Expression");
        if let Ok(_) = token_stream.next_matches(LexToken::OParen) {
            let expression: Expression = token_stream.parse()?;
            token_stream.next_matches(LexToken::CParen)?;
            return Ok(expression);
        }

        let ident: Result<Ident> = token_stream.parse();
        if let Ok(ident) = ident {
            return Ok(Self::Ident(ident));
        }

        let constant = token_stream.next_matches_func(|x| {
            use LexToken::*;
            match x {
                StrLit(x) | IntLit(x) | FloatLit(x) => Some(x.to_string().into_boxed_str()),
                CharLit(x) => Some(format!("{}", x).into_boxed_str()),
                _ => None,
            }
        });

        if let Ok(constant) = constant {
            return Ok(Self::Constant(constant));
        }

        return Err(ParserError::Other);
    }
}

#[derive(Debug)]
pub enum UnaryOperator {
    Not,
}

#[derive(Debug)]
pub enum BinaryOperator {
    EqEq,
}

impl<'a> TryFrom<LexToken<'a>> for BinaryOperator {
    type Error = ParserError<MyLexerError>;

    fn try_from(value: LexToken<'a>) -> Result<Self> {
        todo!()
        // TODO: add this back in when ready
        // match value {
        //     LexToken::Eq => Ok(Self::Eq),
        //     _ => Err(lexer::LexerIteratorError::DoesNotMatch.into()),
        // }
    }
}

impl Parse for BinaryOperator {
    fn from_lexer<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
    ) -> Result<Self> {
        token_stream
            .next_matches_func(|x| BinaryOperator::try_from(x).ok())
            .map_err(|err| err.into())
    }
}
