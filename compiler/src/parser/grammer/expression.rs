use lexer::LexerIterator;

use crate::{
    lexer::{LexToken, MyLexerError},
    parser::{
        safe_parse_wrapper,
        symbol_table::SymbolTable,
        Parse, ParseIntoWith, ParserError, ParserInto, Result,
    },
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

#[inline(always)]
fn binary_op_fun_builder<'a, I, FParser, FCond>(
    parser: FParser,
    cond: FCond,
) -> impl Fn(&mut I, &mut SymbolTable) -> Result<Expression>
where
    I: LexerIterator<'a, LexToken<'a>, MyLexerError>,
    FParser: FnMut(&mut I, &mut SymbolTable) -> Result<Expression> + Clone,
    FCond: Fn(&BinaryOperator) -> Option<()>,
{
    #[inline(always)]
    move |token_stream: &mut I, symbol_table: &mut SymbolTable| {
        let mut parser = parser.clone();
        let expr = safe_parse_wrapper(token_stream, symbol_table, &mut parser)?;

        Ok(token_stream
            .parse_with_many(symbol_table, |token_stream, symbol_table| {
                let meta = token_stream
                    .clone()
                    .next()
                    .ok_or(ParserError::LexerError(
                        lexer::LexerIteratorError::NoMoreTokens,
                    ))??
                    .meta;
                let op: BinaryOperator = token_stream.parse(symbol_table)?;

                cond(&op).ok_or(ParserError::DoesNotMatch(meta.into()))?;

                let second = safe_parse_wrapper(token_stream, symbol_table, &mut parser)?;

                Ok((op, second))
            })
            .map_while(|x| x.ok())
            .fold(expr, |acc, (op, second)| acc.make_binary_op(op, second)))
    }
}

impl Expression {
    pub fn make_binary_op(self, op: BinaryOperator, other: Self) -> Self {
        Self::BinaryOp(Box::new(self), op, Box::new(other))
    }

    fn primary_expression<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
        symbol_table: &mut SymbolTable,
    ) -> Result<Expression> {
        trace!("Entering base expression");

        if let Ok(_) = token_stream.next_matches(LexToken::OParen) {
            let expression: Expression = token_stream.parse(symbol_table)?;
            token_stream.next_matches(LexToken::CParen)?;
            return Ok(expression);
        }

        let ident: Result<Ident> = token_stream.parse(symbol_table);
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
        symbol_table: &mut SymbolTable,
    ) -> Result<Expression> {
        trace!("Entering Non Math Expression");
        let expr = token_stream.parse_with(symbol_table, Self::primary_expression)?;

        Ok(token_stream
            .parse_with_many(symbol_table, |token_stream, symbol_table| {
                let token = token_stream.next_matches_func(|x| {
                    matches!(x, LexToken::OParen | LexToken::Dot | LexToken::OBracket)
                        .then_some(x.clone())
                })?;

                let (op, second) = match token {
                    LexToken::OParen => {
                        let IterPlusError(expr_list, following) = token_stream
                            .parse_with_many(symbol_table, |token_stream, symbol_table| {
                                let expr: Expression = token_stream.parse(symbol_table)?;
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
                        let ident: Ident = token_stream.parse(symbol_table)?;
                        (BinaryOperator::Select, Expression::Ident(ident))
                    }
                    LexToken::OBracket => {
                        let expression: Expression = token_stream.parse(symbol_table)?;
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
        symbol_table: &mut SymbolTable,
    ) -> Result<Expression> {
        let op: Option<UnaryOperator> = token_stream.parse(symbol_table).ok();

        let expr = token_stream.parse_with(symbol_table, Self::postfix_expression)?;

        Ok(match op {
            Some(x) => Expression::UnaryOp(x, Box::new(expr)),
            None => expr,
        })
    }

    fn multiplicative_expression<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
        symbol_table: &mut SymbolTable,
    ) -> Result<Expression> {
        trace!("Entering multiplicative expression");
        use BinaryOperator::*;
        binary_op_fun_builder(Self::unary_expression, |op| match op {
            Div | Mult | Mod => Some(()),
            _ => return None,
        })(token_stream, symbol_table)
    }

    fn additive_expression<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
        symbol_table: &mut SymbolTable,
    ) -> Result<Expression> {
        trace!("Entering additive expression");

        use BinaryOperator::*;
        binary_op_fun_builder(Self::multiplicative_expression, |op| match op {
            Add | Sub => Some(()),
            _ => return None,
        })(token_stream, symbol_table)
    }

    fn shift_expression<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
        symbol_table: &mut SymbolTable,
    ) -> Result<Expression> {
        trace!("Entering Shift Expression");

        use BinaryOperator::*;
        binary_op_fun_builder(Self::additive_expression, |op| match op {
            ShiftRight | ShiftLeft => Some(()),
            _ => return None,
        })(token_stream, symbol_table)
    }

    fn relational_expression<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
        symbol_table: &mut SymbolTable,
    ) -> Result<Expression> {
        trace!("Entering Relational Expression");

        use BinaryOperator::*;
        binary_op_fun_builder(Self::shift_expression, |op| match op {
            Gt | Lt | Gte | Lte => Some(()),
            _ => return None,
        })(token_stream, symbol_table)
    }

    fn equality_expression<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
        symbol_table: &mut SymbolTable,
    ) -> Result<Expression> {
        trace!("Entering Equality Expression");

        use BinaryOperator::*;
        binary_op_fun_builder(Self::relational_expression, |op| match op {
            BoolEq | NotEq => Some(()),
            _ => return None,
        })(token_stream, symbol_table)
    }

    fn bin_and_expression<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
        symbol_table: &mut SymbolTable,
    ) -> Result<Expression> {
        trace!("Entering Binary And Expression");

        use BinaryOperator::*;
        binary_op_fun_builder(Self::equality_expression, |op| match op {
            BitAnd => Some(()),
            _ => return None,
        })(token_stream, symbol_table)
    }

    fn xor_expression<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
        symbol_table: &mut SymbolTable,
    ) -> Result<Expression> {
        trace!("Entering Xor Expression");

        use BinaryOperator::*;
        binary_op_fun_builder(Self::bin_and_expression, |op| match op {
            Xor => Some(()),
            _ => return None,
        })(token_stream, symbol_table)
    }

    fn bin_or_expression<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
        symbol_table: &mut SymbolTable,
    ) -> Result<Expression> {
        trace!("Entering Binary Or Expression");

        use BinaryOperator::*;
        binary_op_fun_builder(Self::xor_expression, |op| match op {
            BitOr => Some(()),
            _ => return None,
        })(token_stream, symbol_table)
    }

    fn log_and_expression<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
        symbol_table: &mut SymbolTable,
    ) -> Result<Expression> {
        trace!("Entering Logical And Expression");

        use BinaryOperator::*;
        binary_op_fun_builder(Self::bin_or_expression, |op| match op {
            LogAnd => Some(()),
            _ => return None,
        })(token_stream, symbol_table)
    }

    fn log_or_expression<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
        symbol_table: &mut SymbolTable,
    ) -> Result<Expression> {
        trace!("Entering Logical Or Expression");

        use BinaryOperator::*;
        binary_op_fun_builder(Self::log_and_expression, |op| match op {
            LogOr => Some(()),
            _ => return None,
        })(token_stream, symbol_table)
    }

    fn assignment_expression<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
        symbol_table: &mut SymbolTable,
    ) -> Result<Expression> {
        token_stream
            .parse_with(symbol_table, |token_stream, symbol_table| {
                let unary = token_stream.parse_with(symbol_table, Self::unary_expression)?;

                let meta = token_stream
                    .clone()
                    .next()
                    .ok_or(ParserError::LexerError(
                        lexer::LexerIteratorError::NoMoreTokens,
                    ))??
                    .meta;
                let op: BinaryOperator = token_stream.parse(symbol_table)?;

                use BinaryOperator::*;
                match op {
                    TIMESEQ | DIVEQ | MODEQ | PLUSEQ | MINUSEQ | SHLEQ | SHREQ | ANDEQ | XOREQ
                    | OREQ | Eq => {}
                    _ => return Err(ParserError::DoesNotMatch(meta.into())),
                };

                let second = token_stream.parse_with(symbol_table, Self::log_or_expression)?;

                Ok(Expression::make_binary_op(unary, op, second))
            })
            .or_else(|_| token_stream.parse_with(symbol_table, Self::log_or_expression))
    }
}

impl Parse for Expression {
    fn from_lexer<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
        symbol_table: &mut SymbolTable,
    ) -> Result<Self> {
        trace!("Entering Expression");
        Self::assignment_expression(token_stream, symbol_table)
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
