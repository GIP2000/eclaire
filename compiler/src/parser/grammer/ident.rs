#[derive(PartialEq)]
pub struct Ident<'a>(&'a str);

impl<'a> crate::parser::Parser<'a> for Ident<'a> {
    type Error = super::Error;

    fn from_lexer<L: crate::lexer::MyLexer<'a>>(lexer: &mut L) -> Result<Self, Self::Error> {
        Ok(Self(lexer.next_matches_func(|&x| match x {
            crate::lexer::LexToken::Ident(x) => Some(x),
            _ => None,
        })?))
    }
}
