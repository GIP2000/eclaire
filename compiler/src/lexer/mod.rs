use anyhow::Result;
use proc_lexer::build_dfa;

#[build_dfa]
#[derive(Debug)]
pub enum LexToken<'a> {
    #[regex("break", func = parse_break)]
    Break,
    #[regex("match", func = parse_break)]
    Match,
    #[regex("=>", func = parse_break)]
    FatArrow,
    #[regex("->", func = parse_break)]
    SkinnyArrow,
    #[regex("if", func = parse_break)]
    If,
    #[regex("else", func = parse_break)]
    Else,
    #[regex("enum", func = parse_break)]
    Enum,
    #[regex("for", func = parse_break)]
    For,
    #[regex("in", func = parse_break)]
    In,
    #[regex("return", func = parse_break)]
    Return,
    #[regex("type", func = parse_break)]
    Type,
    #[regex("while", func = parse_break)]
    While,
    #[regex("loop", func = parse_break)]
    Loop,
    #[regex("+", func = parse_break)]
    Plus,
    #[regex("+=", func = parse_break)]
    PlusEq,
    #[regex("++", func = parse_break)]
    PlusPlus,
    #[regex("-", func = parse_break)]
    Minus,
    #[regex("-=", func = parse_break)]
    MinusEq,
    #[regex("--", func = parse_break)]
    MinusMinus,
    #[regex("/", func = parse_break)]
    Div,
    #[regex("/=", func = parse_break)]
    DivEq,
    #[regex("\\*", func = parse_break)]
    Mult,
    #[regex("\\*=", func = parse_break)]
    MultEq,
    #[regex("%", func = parse_break)]
    Mod,
    #[regex("%=", func = parse_break)]
    ModEq,
    #[regex(">", func = parse_break)]
    Gt,
    #[regex("<", func = parse_break)]
    Lt,
    #[regex(">=", func = parse_break)]
    Gte,
    #[regex("<=", func = parse_break)]
    Lte,
    #[regex("=", func = parse_break)]
    Eq,
    #[regex("==", func = parse_break)]
    EqEq,
    #[regex("!=", func = parse_break)]
    NotEq,
    #[regex("&&", func = parse_break)]
    LogAnd,
    #[regex("\\|\\|", func = parse_break)]
    LogOr,
    #[regex("!", func = parse_break)]
    LogNot,
    #[regex("&", func = parse_break)]
    BitAnd,
    #[regex("&=", func = parse_break)]
    BitAndEq,
    #[regex("\\|", func = parse_break)]
    BitOr,
    #[regex("\\|=", func = parse_break)]
    BitOrEq,
    #[regex("~", func = parse_break)]
    BitNot,
    #[regex("^", func = parse_break)]
    BitXor,
    #[regex("^=", func = parse_break)]
    BitXorEq,
    #[regex("\".*\"", func = parse_break)]
    String(&'a str),
    #[regex("'.*'", func = parse_char)]
    Char(char),
    #[regex("(1|2|3|4|5|6|7|8|9|0)(1|2|3|4|5|6|7|8|9|0)*", func = parse_int)]
    Int(&'a str),
    #[regex("(1|2|3|4|5|6|7|8|9|0)(1|2|3|4|5|6|7|8|9|0)*.(1|2|3|4|5|6|7|8|9|0)*", func = parse_float)]
    Float(&'a str),
    #[regex(".*", func = parse_ident)]
    Ident(&'a str),
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

fn parse_break<'a>(x: &'a str) -> Result<LexToken<'a>> {
    Ok(LexToken::Break)
}

#[derive(Debug)]
pub struct LexTokenData<'a> {
    token: LexToken<'a>,
    line_num: usize,
    col_num: usize,
    raw: &'a str,
}
