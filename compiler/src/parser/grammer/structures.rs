use crate::{
    lexer::LexToken,
    parser::{
        grammer::ident::{Ident, IdentPair},
        symbol_table::SymbolTable,
        Parse, ParseIntoWith, ParserInto, Result,
    },
    utils::iterator::IterPlusError,
};

pub fn block_list<'a>(
    token_stream: &mut impl lexer::LexerIterator<'a, LexToken<'a>, anyhow::Error>,
    symbol_table: &mut SymbolTable,
) -> Result<Vec<IdentPair>> {
    token_stream.next_matches(LexToken::OCBracket)?;

    let IterPlusError(result, following) = token_stream.parse_many(symbol_table).collect();

    token_stream
        .next_matches(LexToken::CCBracket)
        .map_err(|err| following.unwrap_or(err.into()))?;

    Ok(result)
}

pub trait GetSize {
    fn get_size(&self) -> usize;
}

#[derive(Debug, Clone)]
pub struct Struct {
    pub fields: Vec<IdentPair>,
}

impl GetSize for Struct {
    fn get_size(&self) -> usize {
        0
    }
}

impl Parse for Struct {
    fn from_lexer<'a>(
        token_stream: &mut impl lexer::LexerIterator<'a, LexToken<'a>, anyhow::Error>,
        symbol_table: &mut SymbolTable,
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

impl Parse for Enum {
    fn from_lexer<'a>(
        token_stream: &mut impl lexer::LexerIterator<'a, LexToken<'a>, crate::lexer::MyLexerError>,
        symbol_table: &mut SymbolTable,
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
                                let ident: Ident = token_stream.parse(symbol_table)?;
                                _ = token_stream.next_matches(LexToken::Comma);
                                Ok(ident)
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
    Tuple(Vec<Ident>),
    Struct(Vec<IdentPair>),
}

#[derive(Debug, Clone)]
pub struct PrimativeType {
    pub size: usize, // TODO play with this datatype size (should it just be a u16 or a u8?)
    pub like: PrimativeLike,
    pub is_default: bool, // Assert that there is only one of these ever declared in the program.
}

impl Parse for PrimativeType {
    fn from_lexer<'a>(
        token_stream: &mut impl lexer::LexerIterator<'a, LexToken<'a>, crate::lexer::MyLexerError>,
        symbol_table: &mut SymbolTable,
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

        if is_default {
            // TODO: add a check here to see if there is another is_default true that is already in
            // the system for this data type.
        }

        token_stream.next_matches(LexToken::CParen)?;
        Ok(Self {
            size,
            like,
            is_default,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub enum PrimativeLike {
    SInt,
    UInt,
    Float,
}

impl<'a> TryFrom<&LexToken<'a>> for PrimativeLike {
    type Error = ();

    fn try_from(value: &LexToken<'a>) -> std::result::Result<Self, Self::Error> {
        match value {
            LexToken::Int => Ok(PrimativeLike::SInt),
            LexToken::UInt => Ok(PrimativeLike::UInt),
            LexToken::Float => Ok(PrimativeLike::Float),
            _ => Err(()),
        }
    }
}
