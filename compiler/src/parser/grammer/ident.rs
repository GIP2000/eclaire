use lexer::LexerIterator;

use crate::{
    lexer::{LexToken, MyLexerError},
    parser::{Parse, ParserInto, Result},
    trace,
};

#[derive(Debug)]
pub struct Ident {
    pub value: Box<str>,
}

impl<A> PartialEq<A> for Ident
where
    A: AsRef<str>,
{
    fn eq(&self, other: &A) -> bool {
        &*self.value == other.as_ref()
    }
}

impl Parse for Ident {
    fn from_lexer<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
    ) -> Result<Self> {
        trace!("Entering Ident");
        token_stream
            .next_matches_func(|x| {
                if let LexToken::Ident(x) = x {
                    Some(Ident {
                        value: x.to_owned().into(),
                    })
                } else {
                    None
                }
            })
            .map_err(|x| x.into())
    }
}

#[derive(Debug)]
pub struct IdentPair {
    pub name: Ident,
    pub datatype: Ident,
}

impl Parse for IdentPair {
    fn from_lexer<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
    ) -> Result<Self> {
        trace!("Entering Ident Pair");
        let name = token_stream.parse()?;
        token_stream.next_matches(LexToken::Colon)?;
        let datatype = token_stream.parse()?;
        _ = token_stream.next_matches(LexToken::Comma);
        Ok(Self { name, datatype })
    }
}

impl Parse for (Ident, Option<Ident>) {
    fn from_lexer<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
    ) -> Result<Self> {
        token_stream
            .parse()
            .map(|x: IdentPair| (x.name, Some(x.datatype)))
            .or_else(|_| token_stream.parse().map(|x: Ident| (x, None)))
    }
}
