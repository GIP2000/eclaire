use anyhow::Result;
use proc_lexer::Lexer;

#[derive(Debug, Lexer)]
pub enum LexToken<'a> {
    #[regex("break")]
    Break,
    #[regex("match")]
    Match,
    #[regex("=>")]
    FatArrow,
    #[regex("->")]
    SkinnyArrow,
    #[regex("if")]
    If,
    #[regex("else")]
    Else,
    #[regex("enum")]
    Enum,
    #[regex("for")]
    For,
    #[regex("in")]
    In,
    #[regex("return")]
    Return,
    #[regex("type")]
    Type,
    #[regex("while")]
    While,
    #[regex("loop")]
    Loop,
    #[regex("+")]
    Plus,
    #[regex("+=")]
    PlusEq,
    #[regex("++")]
    PlusPlus,
    #[regex("-")]
    Minus,
    #[regex("-=")]
    MinusEq,
    #[regex("--")]
    MinusMinus,
    #[regex("/")]
    Div,
    #[regex("/=")]
    DivEq,
    #[regex("\\*")]
    Mult,
    #[regex("\\*=")]
    MultEq,
    #[regex("%")]
    Mod,
    #[regex("%=")]
    ModEq,
    #[regex(">")]
    Gt,
    #[regex("<")]
    Lt,
    #[regex(">=")]
    Gte,
    #[regex("<=")]
    Lte,
    #[regex("=")]
    Eq,
    #[regex("==")]
    EqEq,
    #[regex("!=")]
    NotEq,
    #[regex("&&")]
    LogAnd,
    #[regex("\\|\\|")]
    LogOr,
    #[regex("!")]
    LogNot,
    #[regex("&")]
    BitAnd,
    #[regex("&=")]
    BitAndEq,
    #[regex("\\|")]
    BitOr,
    #[regex("\\|=")]
    BitOrEq,
    #[regex("~")]
    BitNot,
    #[regex("^")]
    BitXor,
    #[regex("^=")]
    BitXorEq,
    #[regex("\".*\"", func = parse_string)]
    String(&'a str),
    #[regex("'.*'", func = parse_char)]
    Char(char),
    #[regex("[0-9][0-9]*", func = parse_int)]
    Int(&'a str),
    #[regex("[0-9][0-9]*\\.[0-9]*", func = parse_float)]
    Float(&'a str),
    #[regex("[a-zA-Z_][a-zA-Z0-9_]*", func = parse_ident)]
    Ident(&'a str),

    #[regex("\\\\.*\n")]
    #[regex(" ")]
    #[regex("\n")]
    #[regex(";")]
    Skip,
}

fn parse_string<'a>(x: &'a str) -> Result<LexToken<'a>> {
    Ok(LexToken::String(&x[1..(x.len() - 1)]))
}

fn parse_ident<'a>(x: &'a str) -> Result<LexToken<'a>> {
    Ok(LexToken::Ident(x))
}

fn parse_int<'a>(x: &'a str) -> Result<LexToken<'a>> {
    Ok(LexToken::Int(x))
}

fn parse_float<'a>(x: &'a str) -> Result<LexToken<'a>> {
    Ok(LexToken::Float(x))
}

fn parse_char<'a>(x: &'a str) -> Result<LexToken<'a>> {
    Ok(LexToken::Char(
        x.chars()
            .skip(1)
            .next()
            .ok_or(anyhow::anyhow!("Invalid match"))?,
    ))
}
