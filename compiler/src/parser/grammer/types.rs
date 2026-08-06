use std::collections::VecDeque;

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

    Type,
}

#[derive(Debug, PartialEq, FromLexValue)]
#[source(UnaryOperator)]
pub enum TypeIndirection {
    Array(Option<usize>),
    #[left(IntoPointer)]
    Pointer,
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
                Expression::ConstantExpression(ConstantExpression::PrimativeType(typ)) => {
                    ConcreteType::Primative(typ)
                }
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

#[derive(Debug, PartialEq)]
pub enum ConcreteType<'a> {
    Struct(Struct),
    Union(Union),
    Enum(Enum),
    Primative(PrimativeTypes),
    Function(FunctionSig<'a>),
}

#[derive(Debug, PartialEq)]
pub struct Type<'a> {
    indirection: Box<[TypeIndirection]>,
    concrete_type: ConcreteType<'a>,
}

impl<'a> Parser<'a> for Type<'a> {
    type Error = super::Error;

    fn from_lexer<L: crate::lexer::MyLexer<'a>>(lexer: &mut L) -> Result<Self, Self::Error> {
        todo!()
    }
}
