use anyhow::anyhow;

use crate::parser::{
    Parser,
    grammer::statment::{IdentCreation, IdentCreationType},
};

pub mod expression;
pub mod function;
pub mod ident;
pub mod statment;
pub mod structure;
pub mod types;

pub type Error = anyhow::Error;

pub type Result<T> = core::result::Result<T, Error>;

#[derive(Debug)]
pub struct TranslationUnion<'a>(pub Box<[IdentCreation<'a>]>);

impl<'a> Parser<'a> for TranslationUnion<'a> {
    type Error = Error;

    fn from_lexer<L: crate::lexer::MyLexer<'a>>(
        lexer: &mut L,
    ) -> std::result::Result<Self, Self::Error> {
        Ok(Self(
            IdentCreation::parse_many(lexer)
                .map(
                    |x| match x.map_err(|_| anyhow!("No Ident Creation Pattern found")) {
                        r @ Ok(IdentCreation {
                            ic_type: IdentCreationType::Const,
                            ident: _,
                            type_val: _,
                            lvalue: _,
                        }) => r,
                        Ok(_) => Err(anyhow!(" Ident Cration pattern must be Const Type")),
                        Err(err) => Err(err),
                    },
                )
                .collect::<Result<_>>()?,
        ))
    }
}
