pub mod dfa;
mod trie;
mod utils;

use std::{fmt::Debug, marker::PhantomData};

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
    start_pos: usize,
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
            start_pos: self.start_pos.clone(),
            has_errored: self.has_errored.clone(),
            _phantom_data: PhantomData,
        }
    }
}

impl<'a, 'd, A: AcceptFunc, D: DFA<A>> Lex<'a, 'd, A, D> {
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

#[derive(Debug, Error)]
pub enum LexerIteratorError<E> {
    #[error("Match not found")]
    DoesNotMatch,
    #[error("No more tokens to lex")]
    NoMoreTokens,
    #[error(transparent)]
    LexerError(#[from] LexerError<E>),
}

impl<
        'a,
        E: Debug,
        I: Iterator<Item = Result<LexerOutput<'a, T>, LexerIteratorError<E>>> + Clone,
        T: Debug + Clone + PartialEq,
    > LexerIterator<'a, T, E> for I
{
}

pub trait LexerIterator<'a, T: Debug + Clone + PartialEq, E: Debug>:
    Iterator<Item = Result<LexerOutput<'a, T>, LexerIteratorError<E>>> + Clone
{
    fn next_matches(&mut self, rhs: T) -> Result<LexerOutput<'a, T>, LexerIteratorError<E>> {
        let mut other = self.clone();
        let val = other.next().ok_or(LexerIteratorError::NoMoreTokens)??;

        eprintln!("val = {val:?}, rhs = {rhs:?}");

        let result = (val.data == rhs)
            .then_some(val)
            .ok_or(LexerIteratorError::DoesNotMatch)?;

        *self = other;

        eprintln!("next = {:?}", self.clone().next());

        Ok(result)
    }

    fn next_matches_func<R, F: Fn(T) -> Option<R>>(
        &mut self,
        closure: F,
    ) -> Result<R, LexerIteratorError<E>> {
        let mut other = self.clone();
        let val = other.next().ok_or(LexerIteratorError::NoMoreTokens)??;

        eprintln!("val = {val:?}");

        let result = closure(val.data.clone()).ok_or(LexerIteratorError::DoesNotMatch)?;
        *self = other;
        Ok(result)
    }
}

#[derive(Debug)]
pub struct LexerMeta<'a> {
    pub raw_match: &'a str,
    pub index: usize,
}

#[derive(Debug, Error)]
pub enum LexerError<E> {
    #[error("No match found")]
    MatchNotFound,
    #[error("Failed to lex next token")]
    InternalLexerError,
    #[error(transparent)]
    ExternalError(#[from] E),
}

impl<'a, 'd, A, D> Iterator for Lex<'a, 'd, A, D>
where
    A: AcceptFunc,
    D: DFA<A>,
{
    type Item = Result<LexerOutput<'a, A::Output<'a>>, LexerIteratorError<A::Error>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.has_errored || self.start_pos >= self.input.len() {
            return None;
        }

        let (result, new_start) = match self.dfa.get_next_lex(&self.input[self.start_pos..]) {
            Ok(x) => x,
            Err(err) => {
                self.has_errored = true;
                return Some(Err(err.into()));
            }
        };

        // TODO: add lineno and colno
        let meta = LexerMeta {
            raw_match: &self.input[self.start_pos..(self.start_pos + new_start)],
            index: self.start_pos,
        };

        self.start_pos += new_start;

        Some(Ok(LexerOutput { meta, data: result }))
    }
}
