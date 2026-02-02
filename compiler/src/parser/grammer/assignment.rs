use clap::parser::IdsRef;

use crate::{
    debug,
    lexer::LexToken,
    parser::{
        grammer::{
            expression::{ConstantExpression, Expression, TypeDefInfoType},
            ident::Ident,
            structures::PrimativeLike,
        },
        symbol_table::{CompareTypes, SymbolTableType, SymbolTableTypePair},
        Parse, ParseIntoWith, ParserError, ParserInto, Result,
    },
    trace,
};

#[derive(Debug, Clone, Copy)]
pub enum AssignmentType {
    Let(bool),
    Const,
}

impl Parse for AssignmentType {
    fn from_lexer<'a>(
        token_stream: &mut impl lexer::LexerIterator<
            'a,
            crate::lexer::LexToken<'a>,
            crate::lexer::MyLexerError,
        >,
        symbol_table: &mut SymbolTableType,
    ) -> Result<Self> {
        token_stream
            .parse_with(symbol_table, |token_stream, _| {
                token_stream.next_matches(LexToken::Let)?;
                Ok(Self::Let(token_stream.next_matches(LexToken::Mut).is_ok()))
            })
            .or_else(|_| {
                token_stream
                    .next_matches(LexToken::Const)
                    .map(|_| Self::Const)
            })
            .map_err(|err| err.into())
    }
}

#[derive(Debug, Clone, PartialEq, Hash, Eq)]
pub enum TypeRespConcrete {
    IdentRef(Ident),
    Void,
    Pointer(bool, Box<Self>),
}

impl<A: AsRef<str>> PartialEq<A> for TypeRespConcrete {
    fn eq(&self, other: &A) -> bool {
        matches!(self, TypeRespConcrete::IdentRef(x) if x == &other.as_ref())
    }
}

impl From<Ident> for TypeRespConcrete {
    fn from(value: Ident) -> Self {
        Self::IdentRef(value)
    }
}
impl<'a> CompareTypes<'a, TypeRespConcrete> for TypeRespConcrete {
    fn are_types_eq(
        &'a self,
        other: &'a TypeRespConcrete,
        type_defs: (&SymbolTableType, usize),
    ) -> bool {
        match (self, other) {
            (TypeRespConcrete::IdentRef(ident), _) | (_, TypeRespConcrete::IdentRef(ident)) => {
                let a = type_defs.get_until_root(ident);

                match a {
                    Some(a) => a.are_types_eq(other, type_defs),
                    None => false,
                }
            }
            (TypeRespConcrete::Void, TypeRespConcrete::Void) => true,
            (TypeRespConcrete::Void, _) => false,
            (
                TypeRespConcrete::Pointer(is_mut1, type_resp1),
                TypeRespConcrete::Pointer(is_mut2, type_resp2),
            ) => is_mut1 == is_mut2 && type_resp1.are_types_eq(type_resp2.as_ref(), type_defs),
            (TypeRespConcrete::Pointer(_, _), _) => false,
        }
    }
}

impl Parse for TypeRespConcrete {
    fn from_lexer<'a>(
        token_stream: &mut impl lexer::LexerIterator<'a, LexToken<'a>, crate::lexer::MyLexerError>,
        symbol_table: &mut SymbolTableType,
    ) -> Result<Self> {
        let token = token_stream.next_matches_func(|x| match x {
            x @ (LexToken::Ident(_) | LexToken::Ampersand | LexToken::AmpersandAmpersand) => {
                Some(x.clone())
            }
            _ => None,
        })?;

        Ok(match token {
            LexToken::Ident(ident) => Self::IdentRef(ident.into()),

            LexToken::AmpersandAmpersand => {
                let is_mut = token_stream.next_matches(LexToken::Mut).is_ok();
                let next_parse = token_stream.parse(symbol_table).map_err(|err| {
                    debug!("ERROR finding next = {err:?}");
                    err
                })?;
                Self::Pointer(false, Box::new(Self::Pointer(is_mut, Box::new(next_parse))))
            }

            LexToken::Ampersand => {
                let is_mut = token_stream.next_matches(LexToken::Mut).is_ok();
                let next_parse = token_stream.parse(symbol_table).map_err(|err| {
                    debug!("ERROR finding next = {err:?}");
                    err
                })?;
                Self::Pointer(is_mut, Box::new(next_parse))
            }
            _ => unreachable!(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct Assignment {
    pub assignment_type: AssignmentType,
    pub ident: Ident,
    pub data_type: Option<TypeRespConcrete>,
    pub expr: Option<Expression>,
}

impl Assignment {
    pub fn is_const(&self) -> bool {
        use AssignmentType::*;

        match self.assignment_type {
            Const => true,
            Let(_) => false,
        }
    }
}

impl Parse for Assignment {
    fn from_lexer<'a>(
        token_stream: &mut impl lexer::LexerIterator<
            'a,
            crate::lexer::LexToken<'a>,
            crate::lexer::MyLexerError,
        >,
        symbol_table: &mut SymbolTableType,
    ) -> Result<Self> {
        trace!("Entered Assignment");

        let assignment_type: AssignmentType = token_stream.parse(symbol_table)?;

        let (ident, data_type): (Ident, Option<TypeRespConcrete>) =
            token_stream.parse(symbol_table)?;

        let expr: Option<Expression> = token_stream
            .next_matches(LexToken::Eq)
            .ok()
            .map(|_| token_stream.parse(symbol_table))
            .transpose()?;

        use AssignmentType::*;

        match (&assignment_type, expr.as_ref()) {
            (
                Let(_),
                Some(Expression::List(_) | Expression::Constant(ConstantExpression::TypeLit(_))),
            )
            | (Const, None | Some(Expression::List(_))) => return Err(ParserError::Other),

            (Let(_), _) => {}
            (Const, Some(Expression::Constant(ConstantExpression::TypeLit(type_data)))) => {
                match &type_data.type_info {
                    TypeDefInfoType::TypeDefPrim(primative_type) if primative_type.is_default => {
                        match primative_type.like {
                            PrimativeLike::SInt | PrimativeLike::UInt => {
                                assert!(symbol_table.default_int.is_none(), "I checked earlier");

                                symbol_table.default_int = Some(ident.clone());
                            }
                            PrimativeLike::Float => {
                                assert!(symbol_table.default_float.is_none(), "I checked earlier");

                                symbol_table.default_float = Some(ident.clone());
                            }
                            PrimativeLike::Char => {
                                assert!(symbol_table.default_char.is_none(), "I checked earlier");

                                symbol_table.default_char = Some(ident.clone());
                            }
                            PrimativeLike::Bool => {
                                assert!(symbol_table.default_bool.is_none(), "I checked earlier");

                                symbol_table.default_bool = Some(ident.clone());
                            }
                        }
                    }
                    _ => {}
                };
                symbol_table
                    .insert(ident.clone(), type_data.clone())
                    .map_err(|_| ParserError::Other)?;
            }
            (Const, Some(_)) => unimplemented!(),
        }

        token_stream.next_matches(LexToken::SemiColon)?;

        Ok(Self {
            assignment_type,
            ident,
            data_type,
            expr,
        })
    }
}
