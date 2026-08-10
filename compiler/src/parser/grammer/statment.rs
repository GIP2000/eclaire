use crate::{
    lexer::{LexToken, MyLexer},
    parser::{
        Parser, ParserWithState as _,
        grammer::{expression::Expression, function::Function, ident::Ident, types::Type},
    },
    trace,
};

#[derive(Debug, Clone, Copy)]
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
        trace!("Lvalue");

        Function::parse(lexer)
            .map(LValue::Function)
            .or_else(|_| Expression::parse(lexer).map(LValue::Expression))
    }
}

#[derive(Debug)]
pub struct IdentCreation<'a> {
    pub ic_type: IdentCreationType,
    pub ident: Ident<'a>,
    pub type_val: Option<Type<'a>>,
    pub lvalue: LValue<'a>,
}

impl<'a> Parser<'a> for IdentCreation<'a> {
    type Error = super::Error;

    fn from_lexer<L: MyLexer<'a>>(lexer: &mut L) -> Result<Self, Self::Error> {
        trace!("In IdentCreation");
        let ic_type = IdentCreationType::parse(lexer)?;
        let ident = Ident::parse(lexer)?;

        let type_val = lexer
            .next_matches(LexToken::Colon)
            .ok()
            .map(|_| Type::parse(lexer))
            .transpose()?;

        _ = lexer.next_matches(LexToken::Eq)?;

        let lvalue = LValue::parse(lexer)?;

        _ = lexer.next_matches(LexToken::SemiColon)?;

        match (ic_type, lvalue) {
            (ic_type @ IdentCreationType::Let(_), lvalue @ LValue::Expression(_))
            | (ic_type @ IdentCreationType::Const, lvalue) => Ok(Self {
                ic_type,
                ident,
                type_val,
                lvalue,
            }),
            _ => Err(super::Error::DoesNotMatch(
                "Error Let value must have a non-const lvalue",
            )),
        }
    }
}

#[derive(Debug)]
pub enum Statment<'a> {
    IdentCreation(IdentCreation<'a>),
    Expression(super::expression::Expression<'a>),
}

impl<'a> Parser<'a> for Statment<'a> {
    type Error = super::Error;

    fn from_lexer<L: MyLexer<'a>>(lexer: &mut L) -> Result<Self, Self::Error> {
        IdentCreation::parse(lexer)
            .map(Self::IdentCreation)
            .or_else(|_| {
                (|lexer: &mut L| {
                    Expression::parse(lexer).and_then(|e| {
                        lexer
                            .next_matches(LexToken::SemiColon)
                            .map(|_| Self::Expression(e))
                            .map_err(Into::into)
                    })
                })
                .parse(lexer)
            })
    }
}
