use super::*;

use lexer::dfa::DFA;
use LexToken::*;

#[test]
fn test_lexer() {
    let mut lexer = LexToken::lex("fn {}").map(|x| x.map(|(x, _)| x));
    __lexer_gen__::LexTokenDFA.debug_print("asdfn {}\0");

    assert_eq!(lexer.next().unwrap().unwrap(), Fn);
    assert_eq!(lexer.next().unwrap().unwrap(), OCBracket);
    assert_eq!(lexer.next().unwrap().unwrap(), CCBracket);
    assert!(lexer.next().is_none());
}
