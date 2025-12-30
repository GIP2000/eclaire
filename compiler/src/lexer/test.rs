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

#[test]
fn test_intlit() {
    let mut lexer = LexToken::lex("123").map(|x| x.map(|(x, _)| x));
    __lexer_gen__::LexTokenDFA.debug_print("0123456789ab \0");

    assert_eq!(lexer.next().unwrap().unwrap(), IntLit("123"));
}

#[test]
fn test_block_stmt() {
    let mut lexer = LexToken::lex(
        "fn foo() {
        let x: y = 1;
        let y: y = \"asdfasfd\";
        let z: y = 334.234;
        let a: y = 'a';
        let b: y = x;
}",
    )
    .map(|x| x.map(|(x, _)| x));

    assert_eq!(lexer.next().unwrap().unwrap(), Fn);
    assert_eq!(lexer.next().unwrap().unwrap(), Ident("foo"));
    assert_eq!(lexer.next().unwrap().unwrap(), OParen);
    assert_eq!(lexer.next().unwrap().unwrap(), CParen);
    assert_eq!(lexer.next().unwrap().unwrap(), OCBracket);

    assert_eq!(lexer.next().unwrap().unwrap(), Let);
    assert_eq!(lexer.next().unwrap().unwrap(), Ident("x"));
    assert_eq!(lexer.next().unwrap().unwrap(), Colon);
    assert_eq!(lexer.next().unwrap().unwrap(), Ident("y"));
    assert_eq!(lexer.next().unwrap().unwrap(), Eq);
    assert_eq!(lexer.next().unwrap().unwrap(), IntLit("1"));
    assert_eq!(lexer.next().unwrap().unwrap(), SemiColon);

    assert_eq!(lexer.next().unwrap().unwrap(), Let);
    assert_eq!(lexer.next().unwrap().unwrap(), Ident("y"));
    assert_eq!(lexer.next().unwrap().unwrap(), Colon);
    assert_eq!(lexer.next().unwrap().unwrap(), Ident("y"));
    assert_eq!(lexer.next().unwrap().unwrap(), Eq);
    assert_eq!(lexer.next().unwrap().unwrap(), StrLit("asdfasfd"));
    assert_eq!(lexer.next().unwrap().unwrap(), SemiColon);

    assert_eq!(lexer.next().unwrap().unwrap(), Let);
    assert_eq!(lexer.next().unwrap().unwrap(), Ident("z"));
    assert_eq!(lexer.next().unwrap().unwrap(), Colon);
    assert_eq!(lexer.next().unwrap().unwrap(), Ident("y"));
    assert_eq!(lexer.next().unwrap().unwrap(), Eq);
    assert_eq!(lexer.next().unwrap().unwrap(), FloatLit("334.234"));
    assert_eq!(lexer.next().unwrap().unwrap(), SemiColon);

    assert_eq!(lexer.next().unwrap().unwrap(), Let);
    assert_eq!(lexer.next().unwrap().unwrap(), Ident("a"));
    assert_eq!(lexer.next().unwrap().unwrap(), Colon);
    assert_eq!(lexer.next().unwrap().unwrap(), Ident("y"));
    assert_eq!(lexer.next().unwrap().unwrap(), Eq);
    assert_eq!(lexer.next().unwrap().unwrap(), CharLit(b'a'));
    assert_eq!(lexer.next().unwrap().unwrap(), SemiColon);

    assert_eq!(lexer.next().unwrap().unwrap(), Let);
    assert_eq!(lexer.next().unwrap().unwrap(), Ident("b"));
    assert_eq!(lexer.next().unwrap().unwrap(), Colon);
    assert_eq!(lexer.next().unwrap().unwrap(), Ident("y"));
    assert_eq!(lexer.next().unwrap().unwrap(), Eq);
    assert_eq!(lexer.next().unwrap().unwrap(), Ident("x"));
    assert_eq!(lexer.next().unwrap().unwrap(), SemiColon);

    assert_eq!(lexer.next().unwrap().unwrap(), CCBracket);
    assert!(lexer.next().is_none());
}
