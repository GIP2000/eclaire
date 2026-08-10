use crate::lexer::{LexToken, MyLexer};
use crate::parser::grammer::types::{ConcreteType, PrimativeTypes};
use crate::parser::{Parser, ParserWithState};
use crate::trace;
use proc_compiler::FromLexValue;

use super::ident::Ident;
use super::statment::Statment;

macro_rules! chain_binary_op {
    ($name:ident : |$var:ident| $body:block, $next:ident : |$nvar:ident| $nbody:block $($rest:tt)*) => {
        fn $name<L: MyLexer<'a>>(lexer: &mut L) -> super::Result<Self> {
            Self::binary_op_builder(
                Self::$next,
                |&x| {
                    let $var: BinaryOperator = x.try_into().ok()?;
                    $body.then_some($var)
                },
                lexer,
            )
        }

        chain_binary_op!($next : |$nvar| $nbody $($rest)*);
    };

    ($name:ident : |$var:ident| $body:block, $last:ident) => {
        fn $name<L: MyLexer<'a>>(lexer: &mut L) -> super::Result<Self> {
            Self::binary_op_builder(
                Self::$last,
                |&x| {
                    let $var: BinaryOperator = x.try_into().ok()?;
                    $body.then_some($var)
                },
                lexer,
            )
        }
    };

    ($last:ident) => {};
}

#[derive(Debug)]
pub struct BlockExpression<'a>(pub Box<[Statment<'a>]>, pub Box<Expression<'a>>);

impl<'a> Parser<'a> for BlockExpression<'a> {
    type Error = super::Error;

    fn from_lexer<L: MyLexer<'a>>(lexer: &mut L) -> Result<Self, Self::Error> {
        _ = lexer.next_matches(LexToken::OCBracket)?;

        let stmts: Box<_> = Statment::parse_many(lexer).map_while(Result::ok).collect();

        let has_return = lexer.next_matches(LexToken::Return).is_ok();
        let return_expression = Expression::parse(lexer);

        let return_expression = Box::new(match (return_expression, has_return) {
            (Ok(e), true) => {
                _ = lexer.next_matches(LexToken::SemiColon)?;
                e
            }
            (Ok(e), false) => e,
            (Err(_), false) => Expression::ConstantExpression(ConstantExpression::ConcreteType(
                ConcreteType::Primative(PrimativeTypes::Void),
            )),
            (Err(_), true) => {
                return Err(super::Error::DoesNotMatch(
                    "return Specified but no return type expression found",
                ));
            }
        });
        _ = lexer.next_matches(LexToken::CCBracket)?;

        Ok(Self(stmts, return_expression))
    }
}

#[derive(Debug)]
pub enum Expression<'a> {
    ConstantExpression(ConstantExpression<'a>),
    BinaryOp(Box<BinaryOp<'a>>),
    UnaryOp(Box<UnaryExpression<'a>>),
    Ident(Ident<'a>),
    List(Box<[Self]>),
    Block(BlockExpression<'a>),
}

impl<'a> Parser<'a> for Expression<'a> {
    type Error = super::Error;

    fn from_lexer<L: MyLexer<'a>>(lexer: &mut L) -> Result<Self, Self::Error> {
        (|lexer: &mut L| {
            let left = Self::unary_expression.parse(lexer)?;

            let op: AssignmentOperator = lexer.next_matches_func(|&x| {
                let bin_op: BinaryOperator = x.try_into().ok()?;
                bin_op.try_into().ok()
            })?;

            let right = Self::log_or_expression.parse(lexer)?;

            Ok(Expression::BinaryOp(Box::new(BinaryOp {
                left,
                right,
                op: op.into(),
            })))
        })
        .parse(lexer)
        .or_else(|_: Self::Error| Self::log_or_expression.parse(lexer))
    }
}

impl<'a> Expression<'a> {
    fn postfix_expression<L: MyLexer<'a>>(lexer: &mut L) -> Result<Self, super::Error> {
        trace!("postfix_expression");
        let left = Self::primary_expression.parse(lexer)?;

        Ok((|lexer: &mut L| -> super::Result<_> {
            let unvalidated_operator: PostfixOperator = lexer.next_matches_func(|&x| {
                let x: BinaryOperator = x.try_into().ok()?;
                x.try_into().ok()
            })?;

            let right = match unvalidated_operator {
                PostfixOperator::Select => Expression::Ident(Ident::parse(lexer)?),
                PostfixOperator::ArrayIndex => {
                    let result = Expression::parse(lexer)?;
                    lexer.next_matches(LexToken::CBracket)?;
                    result
                }
                PostfixOperator::Call => {
                    let mut must_stop = false;
                    let result: Box<_> = (|lexer: &mut L| -> anyhow::Result<Self> {
                        if must_stop {
                            anyhow::bail!("must stop");
                        }
                        let expr = Expression::parse(lexer)?;

                        if let Err(_) = lexer.next_matches(LexToken::Comma) {
                            must_stop = true;
                        }
                        Ok(expr)
                    })
                    .parse_many(lexer)
                    .map_while(Result::ok)
                    .collect();

                    if !must_stop && result.len() > 0 {
                        return Err(super::Error::DoesNotMatch("Invalid Expression in Call"));
                    }

                    lexer.next_matches(LexToken::CParen)?;
                    Expression::List(result)
                }
            };

            Ok((unvalidated_operator.into(), right))
        })
        .parse_many(lexer)
        .map_while(Result::ok)
        .fold(left, |left, (op, right)| {
            Self::BinaryOp(Box::new(BinaryOp { left, right, op }))
        }))
    }

    fn primary_expression(lexer: &mut impl MyLexer<'a>) -> super::Result<Self> {
        trace!("primary_expression");
        if let Ok(_) = lexer.next_matches(LexToken::OParen) {
            let expr = Expression::parse(lexer)?;
            lexer.next_matches(LexToken::CParen)?;
            return Ok(expr);
        }

        // TODO:  Control Flow when I have statments
        BlockExpression::parse(lexer)
            .map(Self::Block)
            .or_else(|_| ConstantExpression::parse(lexer).map(Self::ConstantExpression))
            .or_else(|_| Ident::parse(lexer).map(Self::Ident))
    }

    fn binary_op_builder<L, NextF, OpF>(
        mut next_f: NextF,
        op_f: OpF,
        lexer: &mut L,
    ) -> super::Result<Expression<'a>>
    where
        L: MyLexer<'a>,
        NextF: ParserWithState<'a, L, Self, Error = super::Error>,
        OpF: Fn(&LexToken<'a>) -> Option<BinaryOperator> + Clone,
    {
        let expr = next_f.parse(lexer)?;

        Ok(
            (|lexer: &mut L| -> super::Result<(BinaryOperator, Expression<'a>)> {
                let op = lexer.next_matches_func(op_f.clone())?;
                let second = next_f.parse(lexer)?;

                Ok((op, second))
            })
            .parse_many(lexer)
            .map_while(|x| x.ok())
            .fold(expr, |left, (op, right)| {
                Self::BinaryOp(Box::new(BinaryOp { left, right, op }))
            }),
        )
    }

    chain_binary_op!(
        log_or_expression: |op| {
            matches!(op, BinaryOperator::LogOr)
        },
        log_and_expression: |op| {
            matches!(op, BinaryOperator::LogAnd)
        },
        bin_or_expression: |op| {
            matches!(op, BinaryOperator::BitOr)
        },
        xor_expression: |op| {
            matches!(op, BinaryOperator::BitOr)
        },
        bin_and_expression: |op| {
            matches!(op, BinaryOperator::BitAnd)
        },
        equality_expression: |op| {
            matches!(op, BinaryOperator::BoolEq | BinaryOperator::NotEq)
        },
        relational_expression: |op| {
            matches!(op, BinaryOperator::Gt | BinaryOperator::Gte | BinaryOperator::Lt | BinaryOperator::Lte)
        },
        shift_expression: |op| {
            matches!(op, BinaryOperator::ShiftLeft | BinaryOperator::ShiftRight)
        },
        additive_expression: |op| {
            matches!(op, BinaryOperator::Add | BinaryOperator::Sub)
        },
        multiplicative_expression: |op| {
            matches!(op, BinaryOperator::Mult | BinaryOperator::Div)
        },
        unary_expression
    );

    pub fn unary_expression<L: MyLexer<'a>>(lexer: &mut L) -> super::Result<Self> {
        trace!("unary_expression");
        let op: Box<_> = UnaryOperator::parse_many(lexer)
            .map_while(Result::ok)
            .collect::<Box<_>>()
            .into_iter()
            .rev()
            .collect();

        let expr = Self::postfix_expression.parse(lexer)?;

        Ok(op.into_iter().fold(expr, |expr, op| {
            Self::UnaryOp(Box::new(UnaryExpression { expr, op }))
        }))
    }
}

#[derive(FromLexValue, Debug, Clone, Copy)]
#[source(LexToken)]
#[generic_impl_override(<'a>)]
#[generic_source_override(<'a>)]
pub enum UnaryOperator {
    #[left(Plus)]
    Pos,
    #[left(Minus)]
    Neg,
    #[left(Bang)]
    Not,
    #[left(Star)]
    FromPointer,
    #[left(Carrot)]
    IntoPointer,
    #[skip]
    Array(Option<usize>),
}

impl<'a> Parser<'a> for UnaryOperator {
    type Error = super::Error;

    fn from_lexer<L: MyLexer<'a>>(lexer: &mut L) -> Result<Self, Self::Error> {
        let unary_op: Result<Self, _> = lexer.next_matches_func(|&x| x.try_into().ok());

        match unary_op {
            Ok(x) => Ok(x),
            Err(_) => {
                lexer.next_matches(LexToken::OBracket)?;
                let size = ConstantExpression::parse(lexer)
                    .and_then(usize::try_from)
                    .ok();
                lexer.next_matches(LexToken::CBracket)?;
                Ok(Self::Array(size))
            }
        }
    }
}

#[derive(FromLexValue, Debug)]
#[source(LexToken)]
#[generic_impl_override(<'a>)]
#[generic_source_override(<'a>)]
pub enum BinaryOperator {
    // multiplicative
    #[left(Star)]
    Mult,
    Div,
    Mod,

    // aditive
    #[left(Plus)]
    Add,
    #[left(Minus)]
    Sub,

    // comparison
    Gt,
    Gte,
    Lt,
    Lte,

    #[left(EqEq)]
    BoolEq,
    NotEq,
    #[left(AmpersandAmpersand)]
    LogAnd,
    #[left(PipePipe)]
    LogOr,

    // Binary
    #[left(Ampersand)]
    BitAnd,
    #[left(Pipe)]
    BitOr,
    ShiftLeft,
    ShiftRight,
    #[left(Carrot)]
    Xor,

    // Selection
    #[left(Dot)]
    Select,
    #[left(OBracket)]
    ArrayIndex,
    #[left(OParen)]
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

#[derive(FromLexValue)]
#[source(BinaryOperator)]
pub enum PostfixOperator {
    Select,
    ArrayIndex,
    Call,
}

#[derive(FromLexValue)]
#[source(BinaryOperator)]
pub enum AssignmentOperator {
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

#[derive(FromLexValue, Debug)]
#[source(LexToken)]
pub enum ConstantExpression<'a> {
    IntLit(&'a str),
    FloatLit(&'a str),
    StrLit(&'a str),
    CharLit(u8),
    BoolLit(bool),

    #[skip]
    ConcreteType(ConcreteType<'a>),
}

impl<'a> TryFrom<ConstantExpression<'a>> for usize {
    type Error = super::Error;

    fn try_from(value: ConstantExpression<'a>) -> Result<Self, Self::Error> {
        use ConstantExpression::*;
        match value {
            IntLit(x) => x
                .parse()
                .map_err(|_| super::Error::DoesNotMatch("Must be a consant integer value")),
            _ => Err(super::Error::DoesNotMatch(
                "Must be a constant integer value",
            )),
        }
    }
}

impl<'a> Parser<'a> for ConstantExpression<'a> {
    type Error = super::Error;

    fn from_lexer<L: MyLexer<'a>>(lexer: &mut L) -> Result<Self, Self::Error> {
        trace!("constant expression");
        Ok(lexer
            .next_matches_func(|&x| x.try_into().ok())
            .or_else(|_| ConcreteType::parse(lexer).map(Self::ConcreteType))?)
    }
}

#[derive(Debug)]
pub struct BinaryOp<'a> {
    pub left: Expression<'a>,
    pub right: Expression<'a>,
    pub op: BinaryOperator,
}

#[derive(Debug)]
pub struct UnaryExpression<'a> {
    pub expr: Expression<'a>,
    pub op: UnaryOperator,
}
