use std::collections::VecDeque;

use crate::parser::ParserWithState;

use proc_compiler::FromLexValue;

use crate::{
    lexer::LexToken,
    parser::{
        Parser,
        grammer::{
            expression::{ConstantExpression, Expression, UnaryExpression, UnaryOperator},
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
        Expression::unary_expression
            .parse(lexer)
            .map(|x| match x {
                Expression::UnaryOp(box unary_expression) => unary_expression,
                _ => unreachable!("I parsed a unary_expression"),
            })
            .and_then(|x| x.try_into())
    }
}

impl<'a> TryFrom<UnaryExpression<'a>> for Type<'a> {
    type Error = super::Error;

    fn try_from(mut value: UnaryExpression<'a>) -> Result<Self, Self::Error> {
        let mut v = VecDeque::new();

        loop {
            let type_indirection: TypeIndirection = value
                .op
                .try_into()
                .map_err(|_| anyhow::anyhow!("Cannot convert Operand to type: {:?}", value.op))?;

            v.push_front(type_indirection);

            let concrete_type = match value.expr {
                Expression::UnaryOp(unary_expression) => {
                    value = *unary_expression;
                    continue;
                }
                Expression::ConstantExpression(ConstantExpression::ConcreteType(typ)) => typ,
                Expression::Ident(_) => todo!("need to implement symbol table"),
                x => anyhow::bail!("Cannot convert Expression to type: {:?} ", x),
            };

            return Ok(Self {
                indirection: Vec::from(v).into_boxed_slice(),
                concrete_type,
            });
        }
    }
}
