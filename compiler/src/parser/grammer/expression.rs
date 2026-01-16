use lexer::LexerIterator;

use crate::{
    lexer::{LexToken, MyLexerError},
    parser::{Parse, ParseIntoWith, ParserError, ParserInto, Result},
    trace,
    utils::iterator::IterPlusError,
};

use super::Ident;

#[derive(Debug)]
pub enum Expression {
    BinaryOp(Box<Expression>, BinaryOperator, Box<Expression>),
    UnaryOp(UnaryOperator, Box<Expression>),

    List(Vec<Expression>),

    Ident(Ident),
    Constant(Box<str>), // TODO: enrich this type / expand it with more stuff
}

impl Expression {
    pub fn make_binary_op(self, op: BinaryOperator, other: Self) -> Self {
        Self::BinaryOp(Box::new(self), op, Box::new(other))
    }
}

fn primary_expression<'a>(
    token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
) -> Result<Expression> {
    trace!("Entering base expression");

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

fn postfix_expression<'a>(
    token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
) -> Result<Expression> {
    trace!("Entering Non Math Expression");
    let expr = token_stream.parse_with(primary_expression)?;

    Ok(token_stream
        .parse_with_many(|token_stream| {
            let token = token_stream.next_matches_func(|x| {
                matches!(x, LexToken::OParen | LexToken::Dot | LexToken::OBracket)
                    .then_some(x.clone())
            })?;

            let (op, second) = match token {
                LexToken::OParen => {
                    let IterPlusError(expr_list, following) = token_stream
                        .parse_with_many(|token_stream| {
                            // this might not be right?
                            let expr: Expression = token_stream.parse()?;
                            _ = token_stream.next_matches(LexToken::Comma);
                            Ok(expr)
                        })
                        .collect();

                    token_stream
                        .next_matches(LexToken::CParen)
                        .map_err(|err| following.unwrap_or(err.into()))?;

                    (BinaryOperator::Call, Expression::List(expr_list))
                }
                LexToken::Dot => {
                    let ident: Ident = token_stream.parse()?;
                    (BinaryOperator::Select, Expression::Ident(ident))
                }
                LexToken::OBracket => {
                    let expression: Expression = token_stream.parse()?;
                    token_stream.next_matches(LexToken::CBracket)?;
                    (BinaryOperator::ArrayIndex, expression)
                }
                _ => unreachable!("I checked for these tokens already"),
            };

            Ok((op, second))
        })
        .map_while(|x| x.ok())
        .fold(expr, |acc, (op, second)| acc.make_binary_op(op, second)))
}

fn unary_expression<'a>(
    token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
) -> Result<Expression> {
    let op: Option<UnaryOperator> = token_stream.parse().ok();

    let expr = token_stream.parse_with(postfix_expression)?;

    Ok(match op {
        Some(x) => Expression::UnaryOp(x, Box::new(expr)),
        None => expr,
    })
}

fn multiplicative_expression<'a>(
    token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
) -> Result<Expression> {
    trace!("Entering multiplicative expression");

    let expr = token_stream.parse_with(unary_expression)?;

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

            use BinaryOperator::*;

            match op {
                Div | Mult | Mod => {}
                _ => return Err(ParserError::DoesNotMatch(meta.into())),
            };

            let second = token_stream.parse_with(unary_expression)?;

            Ok((op, second))
        })
        .map_while(|x| x.ok())
        .fold(expr, |acc, (op, second)| acc.make_binary_op(op, second)))
}

fn additive_expression<'a>(
    token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
) -> Result<Expression> {
    trace!("Entering additive expression");

    let expr = token_stream.parse_with(multiplicative_expression)?;

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

            let second = token_stream.parse_with(multiplicative_expression)?;

            Ok((op, second))
        })
        .map_while(|x| x.ok())
        .fold(expr, |acc, (op, second)| acc.make_binary_op(op, second)))
}

fn shift_expression<'a>(
    token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
) -> Result<Expression> {
    let expr = token_stream.parse_with(additive_expression)?;

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

            use BinaryOperator::*;
            match op {
                ShiftRight | ShiftLeft => {}
                _ => return Err(ParserError::DoesNotMatch(meta.into())),
            };

            let second = token_stream.parse_with(additive_expression)?;

            Ok((op, second))
        })
        .map_while(|x| x.ok())
        .fold(expr, |acc, (op, second)| acc.make_binary_op(op, second)))
}

fn relational_expression<'a>(
    token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
) -> Result<Expression> {
    let expr = token_stream.parse_with(shift_expression)?;

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

            use BinaryOperator::*;
            match op {
                Gt | Lt | Gte | Lte => {}
                _ => return Err(ParserError::DoesNotMatch(meta.into())),
            };

            let second = token_stream.parse_with(shift_expression)?;

            Ok((op, second))
        })
        .map_while(|x| x.ok())
        .fold(expr, |acc, (op, second)| acc.make_binary_op(op, second)))
}

fn equality_expression<'a>(
    token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
) -> Result<Expression> {
    let expr = token_stream.parse_with(relational_expression)?;

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

            use BinaryOperator::*;
            match op {
                BoolEq | NotEq => {}
                _ => return Err(ParserError::DoesNotMatch(meta.into())),
            };

            let second = token_stream.parse_with(relational_expression)?;

            Ok((op, second))
        })
        .map_while(|x| x.ok())
        .fold(expr, |acc, (op, second)| acc.make_binary_op(op, second)))
}

fn bin_and_expression<'a>(
    token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
) -> Result<Expression> {
    let expr = token_stream.parse_with(equality_expression)?;

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

            use BinaryOperator::*;
            match op {
                BitAnd => {}
                _ => return Err(ParserError::DoesNotMatch(meta.into())),
            };

            let second = token_stream.parse_with(equality_expression)?;

            Ok((op, second))
        })
        .map_while(|x| x.ok())
        .fold(expr, |acc, (op, second)| acc.make_binary_op(op, second)))
}

fn xor_expression<'a>(
    token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
) -> Result<Expression> {
    let expr = token_stream.parse_with(bin_and_expression)?;

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

            use BinaryOperator::*;
            match op {
                Xor => {}
                _ => return Err(ParserError::DoesNotMatch(meta.into())),
            };

            let second = token_stream.parse_with(bin_and_expression)?;

            Ok((op, second))
        })
        .map_while(|x| x.ok())
        .fold(expr, |acc, (op, second)| acc.make_binary_op(op, second)))
}

fn bin_or_expression<'a>(
    token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
) -> Result<Expression> {
    let expr = token_stream.parse_with(xor_expression)?;

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

            use BinaryOperator::*;
            match op {
                BitOr => {}
                _ => return Err(ParserError::DoesNotMatch(meta.into())),
            };

            let second = token_stream.parse_with(xor_expression)?;

            Ok((op, second))
        })
        .map_while(|x| x.ok())
        .fold(expr, |acc, (op, second)| acc.make_binary_op(op, second)))
}

fn log_and_expression<'a>(
    token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
) -> Result<Expression> {
    let expr = token_stream.parse_with(bin_or_expression)?;

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

            use BinaryOperator::*;
            match op {
                LogAnd => {}
                _ => return Err(ParserError::DoesNotMatch(meta.into())),
            };

            let second = token_stream.parse_with(bin_or_expression)?;

            Ok((op, second))
        })
        .map_while(|x| x.ok())
        .fold(expr, |acc, (op, second)| acc.make_binary_op(op, second)))
}

fn log_or_expression<'a>(
    token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
) -> Result<Expression> {
    let expr = token_stream.parse_with(log_and_expression)?;

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

            use BinaryOperator::*;
            match op {
                LogOr => {}
                _ => return Err(ParserError::DoesNotMatch(meta.into())),
            };

            let second = token_stream.parse_with(log_and_expression)?;

            Ok((op, second))
        })
        .map_while(|x| x.ok())
        .fold(expr, |acc, (op, second)| acc.make_binary_op(op, second)))
}

fn assignment_expression<'a>(
    token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
) -> Result<Expression> {
    token_stream
        .parse_with(|token_stream| {
            let unary = token_stream.parse_with(unary_expression)?;

            let meta = token_stream
                .clone()
                .next()
                .ok_or(ParserError::LexerError(
                    lexer::LexerIteratorError::NoMoreTokens,
                ))??
                .meta;
            let op: BinaryOperator = token_stream.parse()?;

            use BinaryOperator::*;
            match op {
                TIMESEQ | DIVEQ | MODEQ | PLUSEQ | MINUSEQ | SHLEQ | SHREQ | ANDEQ | XOREQ
                | OREQ | Eq => {}
                _ => return Err(ParserError::DoesNotMatch(meta.into())),
            };

            let second = token_stream.parse_with(log_or_expression)?;

            Ok(Expression::make_binary_op(unary, op, second))
        })
        .or_else(|_| token_stream.parse_with(log_or_expression))
}

impl Parse for Expression {
    fn from_lexer<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
    ) -> Result<Self> {
        trace!("Entering Expression");
        assignment_expression(token_stream)
    }
}

#[derive(Debug)]
pub enum UnaryOperator {
    Pos,
    Neg,

    Not,

    FromPointer,
    IntoPointer,
}

impl<'a> TryFrom<&LexToken<'a>> for UnaryOperator {
    type Error = ();

    fn try_from(value: &LexToken<'a>) -> std::result::Result<Self, Self::Error> {
        match value {
            LexToken::Plus => Ok(Self::Pos),
            LexToken::Minus => Ok(Self::Neg),
            LexToken::Bang => Ok(Self::Not),

            LexToken::Ampersand => Ok(Self::IntoPointer),
            LexToken::Star => Ok(Self::FromPointer),

            _ => Err(()),
        }
    }
}

#[derive(Debug)]
pub enum BinaryOperator {
    // multiplicative
    Mult,
    Div,
    Mod,

    // aditive
    Add,
    Sub,

    // conditional
    Gt,
    Gte,
    Lt,
    Lte,
    BoolEq,
    NotEq,
    LogAnd,
    LogOr,

    // Binary
    BitAnd,
    BitOr,
    ShiftLeft,
    ShiftRight,
    Xor,

    // Selection
    Select,
    ArrayIndex,
    Call,

    // Assignment
    TIMESEQ,
    DIVEQ,
    MODEQ,
    PLUSEQ,
    MINUSEQ,
    SHLEQ,
    SHREQ,
    ANDEQ,
    XOREQ,
    OREQ,
    Eq,
}

impl<'a> TryFrom<&LexToken<'a>> for BinaryOperator {
    type Error = ();

    fn try_from(value: &LexToken<'a>) -> std::result::Result<Self, ()> {
        match value {
            // add sub
            LexToken::Plus => Ok(Self::Add),
            LexToken::Minus => Ok(Self::Sub),

            // mul div mod
            LexToken::Star => Ok(Self::Mult),
            LexToken::Mod => Ok(Self::Mod),
            LexToken::Div => Ok(Self::Div),

            // conditional
            LexToken::Gt => Ok(Self::Gt),
            LexToken::Gte => Ok(Self::Gte),
            LexToken::Lt => Ok(Self::Lt),
            LexToken::Lte => Ok(Self::Lte),
            LexToken::EqEq => Ok(Self::BoolEq),
            LexToken::NotEq => Ok(Self::NotEq),
            LexToken::AmpersandAmpersand => Ok(Self::LogAnd),
            LexToken::PipePipe => Ok(Self::LogOr),

            // bit
            LexToken::Pipe => Ok(Self::BitOr),
            LexToken::Ampersand => Ok(Self::BitAnd),
            LexToken::ShiftLeft => Ok(Self::ShiftLeft),
            LexToken::ShiftRight => Ok(Self::ShiftRight),
            LexToken::Carrot => Ok(Self::Xor),

            // Select
            LexToken::Dot => Ok(Self::Select),

            // Assighment
            LexToken::TIMESEQ => Ok(Self::TIMESEQ),
            LexToken::DIVEQ => Ok(Self::DIVEQ),
            LexToken::MODEQ => Ok(Self::MODEQ),
            LexToken::PLUSEQ => Ok(Self::PLUSEQ),
            LexToken::MINUSEQ => Ok(Self::MINUSEQ),
            LexToken::SHLEQ => Ok(Self::SHLEQ),
            LexToken::SHREQ => Ok(Self::SHREQ),
            LexToken::ANDEQ => Ok(Self::ANDEQ),
            LexToken::XOREQ => Ok(Self::XOREQ),
            LexToken::OREQ => Ok(Self::OREQ),
            LexToken::Eq => Ok(Self::Eq),

            _ => Err(()),
        }
    }
}
