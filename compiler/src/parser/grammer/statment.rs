use crate::{
    lexer::{LexToken, MyLexer},
    parser::{
        Parser,
        grammer::{expression::Expression, function::Function, ident::Ident, types::Type},
    },
};

#[derive(Debug)]
pub enum IdentCreationType {
    Const,
    Let(bool),
}

impl<'a> Parser<'a> for IdentCreationType {
    type Error = super::Error;

    fn from_lexer<L: MyLexer<'a>>(lexer: &mut L) -> Result<Self, Self::Error> {
        let mut val = lexer.next_matches_func(|&x| match x {
            LexToken::Const => Some(IdentCreationType::Const),
            LexToken::Let => Some(IdentCreationType::Let(false)),
            _ => None,
        })?;

        if let IdentCreationType::Let(x) = &mut val {
            if let Ok(_) = lexer.next_matches(LexToken::Mut) {
                *x = true;
            }
        };
        Ok(val)
    }
}

#[derive(Debug)]
pub enum LValue<'a> {
    Expression(super::expression::Expression<'a>),
    Function(super::function::Function<'a>),
}

impl<'a> Parser<'a> for LValue<'a> {
    type Error = super::Error;

    fn from_lexer<L: MyLexer<'a>>(lexer: &mut L) -> Result<Self, Self::Error> {
        Expression::parse(lexer)
            .map(LValue::Expression)
            .or_else(|_| Function::parse(lexer).map(LValue::Function))
    }
}

#[derive(Debug)]
pub enum Statment<'a> {
    IdentCreation(IdentCreationType, Ident<'a>, Option<Type<'a>>, LValue<'a>),
    Expression(super::expression::Expression<'a>),
}

impl<'a> Parser<'a> for Statment<'a> {
    type Error = super::Error;

    fn from_lexer<L: MyLexer<'a>>(lexer: &mut L) -> Result<Self, Self::Error> {
        let ic_type = IdentCreationType::parse(lexer)?;
        let ident = Ident::parse(lexer)?;

        let type_val = lexer
            .next_matches(LexToken::Colon)
            .ok()
            .map(|_| Type::parse(lexer))
            .transpose()?;

        _ = lexer.next_matches(LexToken::Eq)?;

        let lvalue = LValue::parse(lexer)?;

        match (ic_type, lvalue) {
            (ic_type @ IdentCreationType::Let(_), lvalue @ LValue::Expression(_))
            | (ic_type @ IdentCreationType::Const, lvalue) => {
                Ok(Self::IdentCreation(ic_type, ident, type_val, lvalue))
            }
            _ => Err(anyhow::anyhow!(
                "Error Let value must have a non-const lvalue"
            )),
        }
    }
}
