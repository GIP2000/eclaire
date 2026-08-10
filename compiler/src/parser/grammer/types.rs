use std::collections::VecDeque;

use crate::{parser::ParserWithState, trace};

use proc_compiler::FromLexValue;

use crate::{
    lexer::LexToken,
    parser::{
        Parser,
        grammer::{
            expression::{ConstantExpression, Expression, UnaryOperator},
            function::FunctionSig,
            structure::{Enum, Struct, Union},
        },
    },
};

#[derive(FromLexValue, PartialEq, Debug)]
#[source(LexToken)]
#[generic_impl_override(<'a>)]
#[generic_source_override(<'a>)]
pub enum PrimativeTypes {
    U8,
    U16,
    U32,
    U64,
    U128,

    I8,
    I16,
    I32,
    I64,
    I128,

    Void,

    Type,
}

#[derive(Debug, PartialEq, FromLexValue)]
#[source(UnaryOperator)]
pub enum TypeIndirection {
    Array(Option<usize>),
    #[left(IntoPointer)]
    Pointer,
}

#[derive(Debug, PartialEq)]
pub enum ConcreteType<'a> {
    Struct(Struct<'a>),
    Union(Union<'a>),
    Enum(Enum<'a>),
    Primative(PrimativeTypes),
    Function(FunctionSig<'a>),
}

impl<'a> Parser<'a> for ConcreteType<'a> {
    type Error = super::Error;

    fn from_lexer<L: crate::lexer::MyLexer<'a>>(lexer: &mut L) -> Result<Self, Self::Error> {
        trace!("concrete type");
        lexer
            .next_matches_func(|&x| x.try_into().ok())
            .map(Self::Primative)
            .or_else(|_| Ok(Self::Struct(Struct::parse(lexer)?)))
            .or_else(|_: super::Error| Ok(Self::Function(FunctionSig::parse(lexer)?)))
            .or_else(|_: super::Error| Ok(Self::Enum(Enum::parse(lexer)?)))
            .or_else(|_: super::Error| Ok(Self::Union(Union::parse(lexer)?)))
    }
}

#[derive(Debug, PartialEq)]
pub struct Type<'a> {
    indirection: Box<[TypeIndirection]>,
    concrete_type: ConcreteType<'a>,
}

impl<'a> From<ConcreteType<'a>> for Type<'a> {
    fn from(value: ConcreteType<'a>) -> Self {
        Self {
            indirection: Box::new([]),
            concrete_type: value,
        }
    }
}

impl<'a> Parser<'a> for Type<'a> {
    type Error = super::Error;

    fn from_lexer<L: crate::lexer::MyLexer<'a>>(lexer: &mut L) -> Result<Self, Self::Error> {
        trace!("type");
        Expression::unary_expression
            .parse(lexer)
            .and_then(TryInto::try_into)
    }
}

impl<'a> TryFrom<Expression<'a>> for Type<'a> {
    type Error = super::Error;

    fn try_from(mut value: Expression<'a>) -> Result<Self, Self::Error> {
        let mut v = VecDeque::new();

        loop {
            let concrete_type = match value {
                Expression::UnaryOp(unary_expression) => {
                    value = unary_expression.expr;

                    let type_indirection: TypeIndirection =
                        unary_expression.op.try_into().map_err(|_| {
                            super::Error::DoesNotMatch("Cannot convert Operand to type")
                        })?;

                    v.push_front(type_indirection);

                    continue;
                }
                Expression::ConstantExpression(ConstantExpression::ConcreteType(typ)) => typ,
                // Expression::Ident(_) => todo!("need to implement symbol table"),
                _ => return Err(super::Error::DoesNotMatch("Cannot convert Operand to type")),
                // super::Error::DoesNotMatch("Cannot convert Operand to type: {:?}", x.op)
            };

            return Ok(Self {
                indirection: Vec::from(v).into_boxed_slice(),
                concrete_type,
            });
        }
    }
}
