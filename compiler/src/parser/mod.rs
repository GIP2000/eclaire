pub mod grammer;
pub mod symbol_table;

use std::marker::PhantomData;

use crate::lexer::MyLexer;

pub struct ParserIter<'a, 'b, L, V, S>
where
    L: MyLexer<'a>,
    S: ParserWithState<'a, L, V>,
{
    lexer: &'b mut L,
    state: S,
    last_err: bool,
    phantom_data: PhantomData<&'a V>,
}

impl<'a, 'b, L, V, S> Iterator for ParserIter<'a, 'b, L, V, S>
where
    L: MyLexer<'a>,
    S: ParserWithState<'a, L, V>,
{
    type Item = Result<V, S::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.last_err {
            return None;
        }

        let val = self.state.parse(&mut self.lexer);

        if let Err(_) = val {
            self.last_err = true;
        }

        Some(val)
    }
}

fn parse_wrapper<'a, L, V, E, F>(lexer: &mut L, mut f: F) -> Result<V, E>
where
    L: MyLexer<'a>,
    F: FnMut(&mut L) -> Result<V, E>,
{
    let mut obj = lexer.clone();
    let res = f(&mut obj)?;
    *lexer = obj;
    Ok(res)
}

pub trait Parser<'a>
where
    Self: Sized,
{
    type Error;

    fn parse_many<'b, L: MyLexer<'a>>(
        lexer: &'b mut L,
    ) -> ParserIter<'a, 'b, L, Self, impl ParserWithState<'a, L, Self>> {
        ParserIter {
            lexer,
            state: Self::from_lexer,
            last_err: false,
            phantom_data: PhantomData,
        }
    }

    fn parse(lexer: &mut impl MyLexer<'a>) -> Result<Self, Self::Error> {
        parse_wrapper(lexer, Self::from_lexer)
    }

    fn from_lexer<L: MyLexer<'a>>(lexer: &mut L) -> Result<Self, Self::Error>;
}

pub trait ParserWithState<'a, L: MyLexer<'a>, Val> {
    type Error;
    fn parse_many<'b>(
        &mut self,
        lexer: &'b mut L,
    ) -> ParserIter<'a, 'b, L, Val, impl ParserWithState<'a, L, Val>> {
        ParserIter {
            lexer,
            state: |lexer: &mut L| self.from_lexer(lexer),
            last_err: false,
            phantom_data: PhantomData,
        }
    }

    fn parse(&mut self, lexer: &mut L) -> Result<Val, Self::Error> {
        parse_wrapper(lexer, |lexer: &mut L| self.from_lexer(lexer))
    }

    fn from_lexer(&mut self, lexer: &mut L) -> Result<Val, Self::Error>;
}

impl<'a, F, L, Val, E> ParserWithState<'a, L, Val> for F
where
    L: MyLexer<'a>,
    F: FnMut(&mut L) -> Result<Val, E>,
{
    type Error = E;

    fn from_lexer(&mut self, lexer: &mut L) -> Result<Val, Self::Error> {
        (self)(lexer)
    }
}
