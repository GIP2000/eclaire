use anyhow::Result;
use lexer::Lexer;

use crate::lexer::LexToken;

pub fn parse(source_code: &str) -> Result<()> {
    let mut lexer = LexToken::lex(source_code);

    for val in lexer {
        let (data, meta) = val?;
        eprintln!("{:?}: {:?}", meta, data);
    }

    Ok(())
}
