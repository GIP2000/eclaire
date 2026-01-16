pub mod dfa;
mod trie;
mod utils;

use std::{
    fmt::{Debug, Display},
    marker::PhantomData,
};

use dfa::DFA;
use thiserror::Error;

pub trait AcceptFunc: Clone
where
    for<'a> Self::Output<'a>: Debug,
    Self::Error: Debug,
{
    type Error;
    type Output<'a>;
    fn convert<'a>(&self, input: &'a str) -> Result<Self::Output<'a>, Self::Error>;
}

pub trait Lexer<'a, A, D>
where
    A: AcceptFunc,
    D: DFA<A>,
{
    fn lex<'d>(input: &'a str) -> Lex<'a, 'd, A, D>;
}

pub struct Lex<'a, 'd, A, D>
where
    A: AcceptFunc,
    D: DFA<A>,
{
    dfa: &'d D,
    input: &'a str,
    idx: usize,
    lineno: usize,
    colno: usize,
    has_errored: bool,
    _phantom_data: std::marker::PhantomData<A>,
}

impl<'a, 'd, A, D> Clone for Lex<'a, 'd, A, D>
where
    A: AcceptFunc,
    D: DFA<A>,
{
    fn clone(&self) -> Self {
        Self {
            dfa: self.dfa,
            input: self.input,
            idx: self.idx.clone(),
            lineno: self.lineno.clone(),
            colno: self.colno.clone(),
            has_errored: self.has_errored.clone(),
            _phantom_data: PhantomData,
        }
    }
}

impl<'a, 'd, A: AcceptFunc, D: DFA<A>> Lex<'a, 'd, A, D> {
    pub fn new(dfa: &'d D, input: &'a str) -> Self {
        Self {
            dfa,
            input,
            idx: 0,
            lineno: 0,
            colno: 0,
            has_errored: false,
            _phantom_data: PhantomData,
        }
    }
}

pub struct LexerOutput<'a, T> {
    pub meta: LexerMeta<'a>,
    pub data: T,
}

impl<'a, T: Debug> Debug for LexerOutput<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LexerOutput")
            .field("meta", &self.meta)
            .field("data", &self.data)
            .finish()
    }
}

impl<
        'a,
        E: Debug,
        I: Iterator<Item = Result<LexerOutput<'a, T>, LexerIteratorError<E>>> + Clone,
        T: Debug + Clone + PartialEq,
    > LexerIterator<'a, T, E> for I
{
}

pub trait LexerIterator<'a, T: Debug + PartialEq, E: Debug>:
    Iterator<Item = Result<LexerOutput<'a, T>, LexerIteratorError<E>>> + Clone
{
    fn next_matches(&mut self, rhs: T) -> Result<LexerOutput<'a, T>, LexerIteratorError<E>> {
        let mut other = self.clone();
        let val = other.next().ok_or(LexerIteratorError::NoMoreTokens)??;

        let result = if val.data == rhs {
            Ok(val)
        } else {
            Err(LexerIteratorError::DoesNotMatch(val.meta.into()))
        }?;

        *self = other;

        Ok(result)
    }

    fn next_matches_func<R, F: Fn(&T) -> Option<R>>(
        &mut self,
        closure: F,
    ) -> Result<R, LexerIteratorError<E>> {
        let mut other = self.clone();
        let val = other.next().ok_or(LexerIteratorError::NoMoreTokens)??;

        eprintln!("val = {val:?}");

        let result = closure(&val.data).ok_or(LexerIteratorError::DoesNotMatch(val.meta.into()))?;
        *self = other;
        Ok(result)
    }
}

#[derive(Debug, Clone)]
pub struct LexerMeta<'a> {
    pub raw_match: &'a str,
    pub index: usize,
    pub lineno: usize,
    pub colno: usize,
}

#[derive(Debug, Error)]
pub enum LexerError<E> {
    #[error("No token matches input {0}")]
    MatchNotFound(Box<str>),
    #[error(transparent)]
    ExternalError(#[from] E),
}

#[derive(Debug)]
pub struct ErrorMeta {
    pub lineno: usize,
    pub colno: usize,
    pub display: Box<str>,
}

impl ErrorMeta {
    const SIZE: usize = 20;
    pub fn new(lineno: usize, colno: usize, idx: usize, input: &str) -> Self {
        Self {
            lineno,
            colno,
            display: input[idx.saturating_sub(Self::SIZE)..(idx + Self::SIZE).min(input.len())]
                .into(),
        }
    }
}

impl<'a> From<LexerMeta<'a>> for ErrorMeta {
    fn from(value: LexerMeta<'a>) -> Self {
        Self {
            lineno: value.lineno,
            colno: value.colno,
            display: value.raw_match.into(),
        }
    }
}

impl Display for ErrorMeta {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{} \"{}\"", self.lineno, self.colno, self.display)
    }
}

#[derive(Debug, Error)]
pub enum LexerIteratorError<E> {
    #[error("Match not found {0}")]
    DoesNotMatch(ErrorMeta),
    #[error("No more tokens to lex")]
    NoMoreTokens,
    #[error("Error: {1}\n {0}")]
    LexerError(LexerError<E>, ErrorMeta),
}

impl<'a, 'd, A, D> Iterator for Lex<'a, 'd, A, D>
where
    A: AcceptFunc,
    D: DFA<A>,
{
    type Item = Result<LexerOutput<'a, A::Output<'a>>, LexerIteratorError<A::Error>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.has_errored || self.idx >= self.input.len() {
            return None;
        }

        let (result, new_start, lineno, colno) = self.dfa.get_next_lex(&self.input[self.idx..]);

        if lineno > 0 {
            self.lineno += lineno;
            self.colno = colno;
        } else {
            self.colno += colno;
        }

        let result = match result {
            Ok(x) => x,
            Err(err) => {
                self.has_errored = true;
                return Some(Err(LexerIteratorError::LexerError(
                    err,
                    ErrorMeta::new(self.lineno, self.colno, self.idx, &self.input),
                )));
            }
        };

        let meta = LexerMeta {
            raw_match: &self.input[self.idx..(self.idx + new_start)],
            index: self.idx,
            lineno: self.lineno,
            colno: self.colno,
        };

        self.idx += new_start;

        Some(Ok(LexerOutput { meta, data: result }))
    }
}
