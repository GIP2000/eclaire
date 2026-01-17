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

#[derive(Debug)]
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

#[derive(Debug)]
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

#[derive(Debug)]
pub enum EnumVariantTypes {
    None,
    Tuple(Vec<Ident>),
    Struct(Vec<IdentPair>),
}
