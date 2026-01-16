use lexer::LexerIterator;

use crate::{
    lexer::{LexToken, MyLexerError},
    parser::{Parse, ParseIntoWith, ParserError, ParserInto, Result},
    trace,
};

use super::Ident;

#[derive(Debug)]
pub enum Expression {
    BinaryOp(Box<Expression>, BinaryOperator, Box<Expression>),
    UnaryOp(UnaryOperator, Box<Expression>),
    Ident(Ident),
    Constant(Box<str>), // TODO: enrich this type / expand it with more stuff
}

fn factor<'a>(
    token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
) -> Result<Expression> {
    trace!("Entering factor");

    if let Ok(_) = token_stream.next_matches(LexToken::OParen) {
        let expression: Expression = token_stream.parse()?;
        token_stream.next_matches(LexToken::CParen)?;
        return Ok(expression);
    }

    let ident: Result<Ident> = token_stream.parse();
    if let Ok(ident) = ident {
        return Ok(Expression::Ident(ident));
    }

    let constant = token_stream.next_matches_func(|x| {
        use LexToken::*;
        match x {
            StrLit(x) | IntLit(x) | FloatLit(x) => Some(x.to_string().into_boxed_str()),
            CharLit(x) => Some(format!("{}", x).into_boxed_str()),
            _ => None,
        }
    });

    return Ok(Expression::Constant(constant?));
}

fn term<'a>(
    token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
) -> Result<Expression> {
    trace!("Entering term");

    let expr = token_stream.parse_with(factor)?;

    Ok(token_stream
        .parse_with_many(|token_stream| {
            let meta = token_stream
                .clone()
                .next()
                .ok_or(ParserError::LexerError(
                    lexer::LexerIteratorError::NoMoreTokens,
                ))??
                .meta;
            let op: BinaryOperator = token_stream.parse()?;

            match op {
                BinaryOperator::Div | BinaryOperator::Mult => {}
                _ => return Err(ParserError::DoesNotMatch(meta.into())),
            };

            let second = token_stream.parse_with(factor)?;

            Ok((op, second))
        })
        .map_while(|x| x.ok())
        .fold(expr, |acc, (op, second)| {
            Expression::BinaryOp(Box::new(acc), op, Box::new(second))
        }))
}

impl Parse for Expression {
    fn from_lexer<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
    ) -> Result<Self> {
        trace!("Entering Expression");

        let expr = token_stream.parse_with(term)?;

        Ok(token_stream
            .parse_with_many(|token_stream| {
                let meta = token_stream
                    .clone()
                    .next()
                    .ok_or(ParserError::LexerError(
                        lexer::LexerIteratorError::NoMoreTokens,
                    ))??
                    .meta;
                let op: BinaryOperator = token_stream.parse()?;

                match op {
                    BinaryOperator::Add | BinaryOperator::Sub => {}
                    _ => return Err(ParserError::DoesNotMatch(meta.into())),
                };

                let second = token_stream.parse_with(term)?;

                Ok((op, second))
            })
            .map_while(|x| x.ok())
            .fold(expr, |acc, (op, second)| {
                Expression::BinaryOp(Box::new(acc), op, Box::new(second))
            }))
    }
}

#[derive(Debug)]
pub enum UnaryOperator {
    Pos,
    Neg,
}

impl<'a> TryFrom<&LexToken<'a>> for UnaryOperator {
    type Error = ();

    fn try_from(value: &LexToken<'a>) -> std::result::Result<Self, Self::Error> {
        match value {
            LexToken::Plus => Ok(Self::Pos),
            LexToken::Minus => Ok(Self::Neg),
            _ => Err(()),
        }
    }
}

#[derive(Debug)]
pub enum BinaryOperator {
    Mult,
    Div,
    Add,
    Sub,
}

impl<'a> TryFrom<&LexToken<'a>> for BinaryOperator {
    type Error = ();

    fn try_from(value: &LexToken<'a>) -> std::result::Result<Self, ()> {
        match value {
            LexToken::Plus => Ok(Self::Add),
            LexToken::Minus => Ok(Self::Sub),
            LexToken::Mult => Ok(Self::Mult),
            LexToken::Div => Ok(Self::Div),
            _ => Err(()),
        }
    }
}
