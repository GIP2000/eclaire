#[cfg(test)]
mod test;

use crate::{info, trace};
use lexer::LexerOutput;
use proc_lexer::Lexer;

pub type MyLexerError = anyhow::Error;
type Result<T> = std::result::Result<T, MyLexerError>;

#[derive(Debug, Lexer, PartialEq, Clone, Copy)]
#[regex_error(MyLexerError)]
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
    #[regex("const")]
    Const,
    #[regex("=")]
    Eq,
    #[regex(";")]
    SemiColon,
    #[regex(",")]
    Comma,

    #[regex("\\.")]
    Dot,

    #[regex("+")]
    Plus,
    #[regex("-")]
    Minus,
    #[regex("/")]
    Div,
    #[regex("\\*")]
    Star,
    #[regex("%")]
    Mod,

    #[regex(">")]
    Gt,
    #[regex(">=")]
    Gte,
    #[regex("<")]
    Lt,
    #[regex("<=")]
    Lte,
    #[regex("==")]
    EqEq,
    #[regex("!")]
    Bang,
    #[regex("!=")]
    NotEq,

    #[regex("&")]
    Ampersand,
    #[regex("&&")]
    AmpersandAmpersand,

    #[regex("^")]
    Carrot,

    #[regex("\\|")]
    Pipe,
    #[regex("\\|\\|")]
    PipePipe,

    #[regex("<<")]
    ShiftLeft,
    #[regex(">>")]
    ShiftRight,

    #[regex("\\*=")]
    TIMESEQ,
    #[regex("/=")]
    DIVEQ,
    #[regex("%=")]
    MODEQ,
    #[regex("+=")]
    PLUSEQ,
    #[regex("-=")]
    MINUSEQ,
    #[regex("<<=")]
    SHLEQ,
    #[regex(">>=")]
    SHREQ,
    #[regex("&=")]
    ANDEQ,
    #[regex("^=")]
    XOREQ,
    #[regex("\\|=")]
    OREQ,

    #[regex("struct")]
    Struct,
    #[regex("enum")]
    Enum,
    #[regex("primative")]
    Primative,

    #[regex("\"[^\"]*\"", func = parse_string)]
    StrLit(&'a str),
    #[regex("[0-9][0-9]*\\.[0-9]*", func = parse_float)]
    FloatLit(&'a str),
    #[regex("[0-9][0-9]*", func = parse_int)]
    IntLit(&'a str),
    #[regex("'[^\n' ]'", func = parse_char)]
    CharLit(u8),
    #[regex("[a-zA-Z_][a-zA-Z0-9_]*", func = parse_ident)]
    Ident(&'a str),

    #[regex("[ \n\t]")]
    #[regex("//[^\n]*\n")]
    Skip,
}

impl<'a> LexToken<'a> {
    pub fn lex(input: &'a str) -> impl lexer::LexerIterator<'a, Self, MyLexerError> {
        <Self as lexer::Lexer<'a, _, _>>::lex(input).filter(|x| match x {
            Ok(LexerOutput {
                meta: _,
                data: LexToken::Skip,
            }) => false,
            x => {
                trace!("lexed a token");
                info!("token found: {:?}", x);
                true
            }
        })
    }
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
