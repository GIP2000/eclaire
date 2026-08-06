use crate::parser::{
    Parser,
    grammer::{expression::BlockExpression, ident::Ident},
};

#[derive(Debug, PartialEq)]
pub struct FunctionSig<'a> {
    args: Box<[Ident<'a>]>,
}

#[derive(Debug)]
pub struct Function<'a> {
    sig: FunctionSig<'a>,
    block: BlockExpression<'a>,
}

impl<'a> Parser<'a> for Function<'a> {
    type Error = super::Error;

    fn from_lexer<L: crate::lexer::MyLexer<'a>>(lexer: &mut L) -> Result<Self, Self::Error> {
        todo!()
    }
}
