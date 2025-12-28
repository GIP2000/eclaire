pub mod dfa;
mod trie;

use std::marker::PhantomData;

use dfa::DFA;

pub trait AcceptFunc
where
    for<'a> Self::Output<'a>: std::fmt::Debug,
{
    type Output<'a>;
    fn convert<'a>(&self, input: &'a str) -> anyhow::Result<Self::Output<'a>>;
}

pub trait Lexer<'a, A, D>
where
    A: AcceptFunc + Clone,
    D: DFA<A>,
{
    fn lex<'d>(input: &'a str) -> Lex<'a, 'd, A, D>;
}

pub struct Lex<'a, 'd, A, D>
where
    A: AcceptFunc + Clone,
    D: DFA<A>,
{
    dfa: &'d D,
    input: &'a str,
    start_pos: usize,
    has_errored: bool,
    _phantom_data: std::marker::PhantomData<A>,
}

impl<'a, 'd, A: AcceptFunc + Clone, D: DFA<A>> Lex<'a, 'd, A, D> {
    pub fn new(dfa: &'d D, input: &'a str, start_pos: usize, has_errored: bool) -> Self {
        Self {
            dfa,
            input,
            start_pos,
            has_errored,
            _phantom_data: PhantomData,
        }
    }
}

impl<'a, 'd, A, D> Iterator for Lex<'a, 'd, A, D>
where
    A: AcceptFunc + Clone,
    D: DFA<A>,
{
    type Item = anyhow::Result<A::Output<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.has_errored || self.start_pos >= self.input.len() {
            return None;
        }

        let (result, new_start) = match self.dfa.get_next_lex(&self.input[self.start_pos..]) {
            Ok(x) => x,
            Err(err) => {
                self.has_errored = true;
                return Some(Err(anyhow::anyhow!(
                    "Failed to lex the next token: {err:?}"
                )));
            }
        };

        self.start_pos += new_start;
        Some(Ok(result))
    }
}
