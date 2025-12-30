#[cfg(test)]
mod test;

use crate::lexer::{LexToken, LexerIterator};

use thiserror::Error;

use LexToken::*;

#[derive(Debug, Error)]
pub enum ParserError {
    #[error("The pattern could not be started")]
    EntryError,
    #[error("There was an error when parsing")]
    MiddleError,
}

pub type Result<T> = core::result::Result<T, ParserError>;

impl From<anyhow::Error> for ParserError {
    fn from(_value: anyhow::Error) -> Self {
        ParserError::MiddleError
    }
}

pub trait FromParseResultTrait<T> {
    fn to_entry(self) -> Result<T>;
}

impl<T> FromParseResultTrait<T> for anyhow::Result<T> {
    fn to_entry(self) -> Result<T> {
        self.map_err(|_| ParserError::EntryError)
    }
}

impl<T> FromParseResultTrait<T> for Result<T> {
    fn to_entry(self) -> Result<T> {
        self.map_err(|_| ParserError::EntryError)
    }
}

fn parse_translation_unit<'a>(lexer: &mut impl LexerIterator<'a>) -> Result<()> {
    let mut first = true;
    loop {
        match (first, parse_function(lexer)) {
            (_, x @ Err(ParserError::MiddleError)) | (true, x @ Err(_)) => {
                eprintln!("I was in middle or first");
                return x;
            }
            (false, Err(ParserError::EntryError)) => {
                eprintln!("I was in entry error");
                break;
            }
            (_, Ok(_)) => {
                eprintln!("I was in OK");
            }
        }

        first = false;
    }

    Ok(())
}

fn ident_type_pair<'a>(lexer: &mut impl LexerIterator<'a>) -> Result<()> {
    eprintln!("I entered the pair");
    let mut lex = lexer.clone();

    lex.next_matches_func(|x| matches!(x, Ident(_)))
        .to_entry()?;

    lex.next_matches(Colon)?;
    lex.next_matches_func(|x| matches!(x, Ident(_)))?;

    *lexer = lex;
    Ok(())
}

fn expression<'a>(lexer: &mut impl LexerIterator<'a>) -> Result<()> {
    let mut lex = lexer.clone();

    lex.next_matches_func(|x| {
        matches!(
            x,
            Ident(_) | CharLit(_) | StrLit(_) | IntLit(_) | FloatLit(_)
        )
    })
    .to_entry()?;

    *lexer = lex;
    Ok(())
}

fn variable_decl<'a>(lexer: &mut impl LexerIterator<'a>) -> Result<()> {
    let mut lex = lexer.clone();

    ident_type_pair(&mut lex).to_entry()?;

    lex.next_matches(Eq)?;

    expression(&mut lex)?;

    lex.next_matches(SemiColon)?;

    *lexer = lex;
    Ok(())
}

fn block_stmt<'a>(lexer: &mut impl LexerIterator<'a>) -> Result<()> {
    let mut lex = lexer.clone();

    lex.next_matches(OCBracket).to_entry()?;
    eprintln!("this is after the OCBracket");

    loop {
        match variable_decl(&mut lex) {
            x @ Err(ParserError::MiddleError) => {
                return x;
            }
            Err(ParserError::EntryError) => break,
            Ok(_) => continue,
        }
    }

    lex.next_matches(CCBracket)?;
    *lexer = lex;
    Ok(())
}

fn parse_function<'a>(lexer: &mut impl LexerIterator<'a>) -> Result<()> {
    let mut lex = lexer.clone();

    lex.next_matches(Fn).to_entry()?;
    eprintln!("found fn");
    lex.next_matches_func(|x| matches!(x, Ident(_)))?;
    eprintln!("found ident");
    lex.next_matches(OParen)?;
    eprintln!("found OParen");

    loop {
        match ident_type_pair(lexer) {
            x @ Err(ParserError::MiddleError) => {
                return x;
            }
            Err(ParserError::EntryError) => break,
            Ok(_) => continue,
        }
    }

    eprintln!("I left the pair");

    lex.next_matches(CParen)?;

    if let Ok(_) = lex.next_matches(SkinnyArrow) {
        lex.next_matches_func(|x| matches!(x, Ident(_)))?;
    }

    block_stmt(&mut lex)?;

    *lexer = lex;
    Ok(())
}

pub fn parse(source_code: &str) -> Result<()> {
    let mut lexer = LexToken::lex(source_code);
    let lexemes: Box<_> = lexer.clone().collect();
    eprintln!("lexemes: {:?}", lexemes);

    parse_translation_unit(&mut lexer)?;

    Ok(())
}
