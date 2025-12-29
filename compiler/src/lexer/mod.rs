use anyhow::Result;
use proc_lexer::Lexer;

#[derive(Debug, Lexer)]
pub enum LexToken<'a> {
    #[regex("function")]
    Fn,
    #[regex("\\(")]
    RParen,
    #[regex("\\)")]
    LParen,
    #[regex(":")]
    Colon,
    #[regex("->")]
    SkinnyArrow,
    #[regex("{")]
    RCBracket,
    #[regex("}")]
    LCBracket,
    #[regex("let")]
    Let,
    #[regex("=")]
    Eq,

    #[regex("[a-zA-Z_][a-zA-Z0-9_]*", func = parse_ident)]
    Ident(&'a str),
    #[regex("\"[^\"]*\"", func = parse_string)]
    StrLit(&'a str),
    #[regex("[0-9][0-9]*", parse_int)]
    IntLit(&'a str),
    #[regex("[0-9][0-9]*\\.[0-9]*", parse_float)]
    FloatLit(&'a str),
    #[regex("'[^\n' ]'", func = parse_char)]
    CharLit(u8),

    #[regex("[ \n\t]")]
    #[regex("//[^\n]*\n")]
    Skip,
}

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
