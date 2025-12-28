pub mod dfa;
mod trie;

use dfa::DFA;

pub trait AcceptFunc
where
    for<'a> Self::Output<'a>: std::fmt::Debug,
{
    type Output<'a>;
    fn convert<'a>(&self, input: &'a str) -> anyhow::Result<Self::Output<'a>>;
}

pub trait Lexer<'a, D>
where
    D: DFA,
{
    fn lex<'d>(input: &'a str) -> Lex<'a, 'd, D>;
}

pub struct Lex<'a, 'd, D>
where
    D: DFA,
{
    dfa: &'d D,
    input: &'a str,
    start_pos: usize,
    has_errored: bool,
}

impl<'a, 'd, D: DFA> Lex<'a, 'd, D> {
    pub fn new(dfa: &'d D, input: &'a str, start_pos: usize, has_errored: bool) -> Self {
        Self {
            dfa,
            input,
            start_pos,
            has_errored,
        }
    }
}

impl<'a, 'd, D> Iterator for Lex<'a, 'd, D>
where
    D: DFA,
{
    type Item = anyhow::Result<D::M<'a>>;

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
