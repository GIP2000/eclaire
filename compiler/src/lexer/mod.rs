#[cfg(test)]
mod test;

use anyhow::{anyhow, Result};
use proc_lexer::Lexer;

#[derive(Debug, Lexer, PartialEq, Clone, Copy)]
pub enum LexToken<'a> {
    #[regex("fn")]
    Fn,
    #[regex("\\(")]
    OParen,
    #[regex("\\)")]
    CParen,
    #[regex(":")]
    Colon,
    #[regex("->")]
    SkinnyArrow,
    #[regex("{")]
    OCBracket,
    #[regex("}")]
    CCBracket,
    #[regex("\\[")]
    OBracket,
    #[regex("\\]")]
    CBracket,
    #[regex("let")]
    Let,
    #[regex("=")]
    Eq,
    #[regex(";")]
    SemiColon,

    #[regex("\"[^\"]*\"", func = parse_string)]
    StrLit(&'a str),
    #[regex("[0-9][0-9]*", func = parse_int)]
    IntLit(&'a str),
    #[regex("[0-9][0-9]*\\.[0-9]*", func = parse_float)]
    FloatLit(&'a str),
    #[regex("'[^\n' ]'", func = parse_char)]
    CharLit(u8),
    #[regex("[a-zA-Z_][a-zA-Z0-9_]*", func = parse_ident)]
    Ident(&'a str),

    #[regex("[ \n\t]")]
    #[regex("//[^\n]*\n")]
    Skip,
}

pub trait LexerIterator<'a>: std::iter::Iterator<Item = LexerOutput<'a>> + Clone {
    fn next_matches(&mut self, rhs: LexToken<'a>) -> LexerOutput<'a> {
        let mut other = self.clone();
        let val = other
            .next()
            .ok_or(anyhow!("No more lexical tokens found"))??;

        eprintln!("val = {val:?}, rhs = {rhs:?}");

        let result = (val.0 == rhs).then_some(val).ok_or(anyhow!("No match"))?;

        *self = other;

        eprintln!("next = {:?}", self.clone().next());

        Ok(result)
    }

    fn next_matches_func<F: Fn(LexToken<'a>) -> bool>(&mut self, closure: F) -> LexerOutput<'a> {
        let mut other = self.clone();
        let val = other
            .next()
            .ok_or(anyhow!("No more lexical tokens found"))??;

        eprintln!("val = {val:?}");

        let result = closure(val.0).then_some(val).ok_or(anyhow!("No match"))?;
        *self = other;
        Ok(result)
    }
}
impl<'a, I> LexerIterator<'a> for I where I: std::iter::Iterator<Item = LexerOutput<'a>> + Clone {}

impl<'a> LexToken<'a> {
    pub fn lex(input: &'a str) -> impl LexerIterator<'a> {
        <Self as lexer::Lexer<'a, _, _>>::lex(input)
            .filter(|x| !matches!(x, Ok((LexToken::Skip, _))))
    }
}

pub type LexerOutput<'a> = <__lexer_gen__::LexerType<'a> as std::iter::Iterator>::Item;

fn parse_string<'a>(x: &'a str) -> Result<LexToken<'a>> {
    Ok(LexToken::StrLit(&x[1..(x.len() - 1)]))
}

fn parse_ident<'a>(x: &'a str) -> Result<LexToken<'a>> {
    Ok(LexToken::Ident(x))
}

fn parse_int<'a>(x: &'a str) -> Result<LexToken<'a>> {
    Ok(LexToken::IntLit(x))
}

fn parse_float<'a>(x: &'a str) -> Result<LexToken<'a>> {
    Ok(LexToken::FloatLit(x))
}

fn parse_char<'a>(x: &'a str) -> Result<LexToken<'a>> {
    Ok(LexToken::CharLit(
        x.bytes()
            .skip(1)
            .next()
            .ok_or(anyhow::anyhow!("Invalid match"))?,
    ))
}
