use proc_compiler::FromLexValue;

use crate::{
    lexer::LexToken,
    parser::{
        Parser,
        grammer::{
            function::FunctionSig,
            structure::{Enum, Struct, Union},
        },
    },
};

#[derive(FromLexValue, PartialEq)]
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

#[derive(PartialEq)]
pub enum TypeIndirection {
    Array(Option<usize>),
    Pointer,
}

#[derive(PartialEq)]
pub enum ConcreteType<'a> {
    Struct(Struct),
    Union(Union),
    Enum(Enum),
    Primative(PrimativeTypes),
    Function(FunctionSig<'a>),
}

#[derive(PartialEq)]
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
