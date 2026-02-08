use crate::{
    lexer::LexToken,
    parser::{
        grammer::{
            assignment::TypeRespConcrete,
            expression::TypeResp,
            ident::{Ident, IdentPair},
        },
        symbol_table::{CompareTypes, STTIdxPair, SymbolTableType},
        Parse, ParseIntoWith, ParserInto, Result,
    },
    utils::iterator::IterPlusError,
};

pub fn block_list<'a>(
    token_stream: &mut impl lexer::LexerIterator<'a, LexToken<'a>, anyhow::Error>,
    symbol_table: &mut SymbolTableType,
) -> Result<Vec<IdentPair>> {
    token_stream.next_matches(LexToken::OCBracket)?;

    let IterPlusError(result, following) = token_stream.parse_many(symbol_table).collect();

    token_stream
        .next_matches(LexToken::CCBracket)
        .map_err(|err| following.unwrap_or(err.into()))?;

    Ok(result)
}

#[derive(Debug, Clone)]
pub struct Struct {
    pub fields: Vec<IdentPair>,
}

impl<'a> CompareTypes<'a, Struct> for Struct {
    fn are_types_eq(&'a self, other: &'a Self, type_defs: STTIdxPair<'_>) -> bool {
        self.fields
            .iter()
            .zip(other.fields.iter())
            .all(|(field_a, field_b)| {
                field_a.name == field_b.name
                    && field_a.datatype.are_types_eq(&field_b.datatype, type_defs)
            })
    }
}

impl Parse for Struct {
    fn from_lexer<'a>(
        token_stream: &mut impl lexer::LexerIterator<'a, LexToken<'a>, anyhow::Error>,
        symbol_table: &mut SymbolTableType,
    ) -> Result<Self> {
        token_stream.next_matches(LexToken::Struct)?;

        token_stream
            .parse_with(symbol_table, block_list)
            .map(|fields| Self { fields })
    }
}

#[derive(Debug, Clone)]
pub struct Enum {
    pub variants: Vec<(Ident, EnumVariantTypes)>,
}

impl<'a> CompareTypes<'a, Enum> for Enum {
    fn are_types_eq(&'a self, other: &'a Self, type_defs: STTIdxPair<'_>) -> bool {
        self.variants.iter().zip(other.variants.iter()).all(
            |((ident_a, variant_type_a), (ident_b, variant_type_b))| {
                if ident_a != ident_b {
                    return false;
                }

                use EnumVariantTypes::*;

                match (variant_type_a, variant_type_b) {
                    (Tuple(t1), Tuple(t2)) => t1
                        .iter()
                        .zip(t2.iter())
                        .all(|(type_a, type_b)| type_a.are_types_eq(type_b, type_defs)),
                    (Struct(s1), Struct(s2)) => s1.iter().zip(s2.iter()).all(|(a, b)| {
                        if a.name != b.name {
                            return false;
                        }
                        a.datatype.are_types_eq(&b.datatype, type_defs)
                    }),
                    (None, None) => true,
                    _ => false,
                }
            },
        )
    }
}

impl Parse for Enum {
    fn from_lexer<'a>(
        token_stream: &mut impl lexer::LexerIterator<'a, LexToken<'a>, crate::lexer::MyLexerError>,
        symbol_table: &mut SymbolTableType,
    ) -> Result<Self> {
        token_stream.next_matches(LexToken::Enum)?;
        token_stream.next_matches(LexToken::OCBracket)?;

        let IterPlusError(variants, following) = token_stream
            .parse_with_many(symbol_table, |token_stream, symbol_table| {
                let variant_name: Ident = token_stream.parse(symbol_table)?;

                let variant_type = token_stream
                    .parse_with(symbol_table, |token_stream, symbol_table| {
                        token_stream.next_matches(LexToken::OParen)?;

                        let IterPlusError(tuple_types, following) = token_stream
                            .parse_with_many(symbol_table, |token_stream, symbol_table| {
                                let type_name = token_stream.parse(symbol_table)?;
                                _ = token_stream.next_matches(LexToken::Comma);
                                Ok(type_name)
                            })
                            .collect();

                        token_stream
                            .next_matches(LexToken::CParen)
                            .map_err(|err| following.unwrap_or(err.into()))?;

                        Ok(EnumVariantTypes::Tuple(tuple_types))
                    })
                    .or_else(|_| {
                        token_stream
                            .parse_with(symbol_table, block_list)
                            .map(|x| EnumVariantTypes::Struct(x))
                    })
                    .unwrap_or(EnumVariantTypes::None);

                _ = token_stream.next_matches(LexToken::Comma);

                Ok((variant_name, variant_type))
            })
            .collect();

        token_stream
            .next_matches(LexToken::CCBracket)
            .map_err(|err| following.unwrap_or(err.into()))?;

        Ok(Self { variants })
    }
}

#[derive(Debug, Clone)]
pub enum EnumVariantTypes {
    None,
    Tuple(Vec<TypeRespConcrete>),
    Struct(Vec<IdentPair>),
}

#[derive(Debug, Clone)]
pub struct PrimativeType {
    pub size: usize, // TODO play with this datatype size (should it just be a u16 or a u8?)
    pub like: PrimativeLike,
    pub is_default: bool, // Assert that there is only one of these ever declared in the program.
}

impl PartialEq<PrimativeType> for PrimativeType {
    fn eq(&self, other: &PrimativeType) -> bool {
        self.size == other.size && self.like == other.like
    }
}

impl Parse for PrimativeType {
    fn from_lexer<'a>(
        token_stream: &mut impl lexer::LexerIterator<'a, LexToken<'a>, crate::lexer::MyLexerError>,
        symbol_table: &mut SymbolTableType,
    ) -> Result<Self> {
        token_stream.next_matches(LexToken::Primative)?;
        token_stream.next_matches(LexToken::OParen)?;
        let like: PrimativeLike = token_stream.parse(symbol_table)?;

        token_stream.next_matches(LexToken::Comma)?;

        let size = token_stream.next_matches_func(|x| match x {
            LexToken::IntLit(x) => Some(x.parse::<usize>().expect("This should always work")),
            _ => None,
        })?;

        if let PrimativeLike::Float = like {
            if size != 32 && size != 64 {
                // TODO: add a new error message for this specificlly
                return Err(crate::parser::ParserError::Other);
            }
        }

        let is_default = token_stream
            .parse_with(symbol_table, |token_stream, _| {
                token_stream.next_matches(LexToken::Comma)?;
                let val = token_stream.next_matches_func(|x| match x {
                    LexToken::BoolLit(x) => Some(*x),
                    _ => None,
                })?;
                Ok(val)
            })
            .unwrap_or(false);

        match (is_default, like) {
            (true, PrimativeLike::SInt | PrimativeLike::UInt) => {
                if symbol_table.default_int.is_some() {
                    return Err(crate::parser::ParserError::Other);
                }
            }
            (true, PrimativeLike::Float) => {
                if symbol_table.default_float.is_some() {
                    return Err(crate::parser::ParserError::Other);
                }
            }
            (true, PrimativeLike::Bool) => {
                if symbol_table.default_bool.is_some() {
                    return Err(crate::parser::ParserError::Other);
                }
            }
            (true, PrimativeLike::Char) => {
                if symbol_table.default_char.is_some() {
                    return Err(crate::parser::ParserError::Other);
                }
            }
            (false, _) => {}
        }

        token_stream.next_matches(LexToken::CParen)?;
        Ok(Self {
            size,
            like,
            is_default,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PrimativeLike {
    SInt,
    UInt,
    Float,
    Char,
    Bool,
}

impl From<PrimativeLike> for TypeResp {
    fn from(value: PrimativeLike) -> Self {
        match value {
            PrimativeLike::UInt | PrimativeLike::SInt => TypeResp::IntLike,
            PrimativeLike::Float => TypeResp::FloatLike,
            PrimativeLike::Char => TypeResp::CharLike,
            PrimativeLike::Bool => TypeResp::BoolLike,
        }
    }
}

impl<'a> TryFrom<&LexToken<'a>> for PrimativeLike {
    type Error = ();

    fn try_from(value: &LexToken<'a>) -> std::result::Result<Self, Self::Error> {
        match value {
            LexToken::Int => Ok(PrimativeLike::SInt),
            LexToken::UInt => Ok(PrimativeLike::UInt),
            LexToken::Float => Ok(PrimativeLike::Float),
            LexToken::Bool => Ok(PrimativeLike::Bool),
            LexToken::Char => Ok(PrimativeLike::Char),
            _ => Err(()),
        }
    }
}
