use std::marker::PhantomData;

use anyhow::Result;

pub type ConversionFn<M> = fn(&str) -> Result<M>;

#[derive(Copy)]
pub enum TransitionType<M: std::fmt::Debug> {
    Normal(usize),
    Fail,
    Accpet(ConversionFn<M>),
    AccpetOr(usize, ConversionFn<M>),
}

impl<M: std::fmt::Debug> std::fmt::Debug for TransitionType<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Normal(arg0) => f.debug_tuple("Normal").field(arg0).finish(),
            Self::Fail => write!(f, "Fail"),
            Self::Accpet(_) => write!(f, "Accept"),
            Self::AccpetOr(arg0, _) => f.debug_tuple("AccpetOr").field(arg0).finish(),
        }
    }
}

impl<M: std::fmt::Debug> TransitionType<M> {
    pub fn upgrade(&mut self, f: ConversionFn<M>) {
        use TransitionType::*;

        match self {
            Normal(x) => *self = AccpetOr(*x, f),
            Fail => *self = Accpet(f),
            _ => {}
        };
    }

    pub fn add_value(&mut self, value: usize) {
        *self = match self {
            TransitionType::Normal(_) | TransitionType::Fail => TransitionType::Normal(value),
            TransitionType::Accpet(f) | TransitionType::AccpetOr(_, f) => {
                TransitionType::AccpetOr(value, f.clone())
            }
        }
    }

    pub fn is_accpet(&self) -> bool {
        match self {
            TransitionType::Normal(_) | TransitionType::Fail => false,
            TransitionType::Accpet(_) | TransitionType::AccpetOr(_, _) => true,
        }
    }
}

impl<M: std::fmt::Debug> Clone for TransitionType<M> {
    fn clone(&self) -> Self {
        match self {
            Self::Normal(arg0) => Self::Normal(arg0.clone()),
            Self::Fail => Self::Fail,
            Self::Accpet(arg0) => Self::Accpet(arg0.clone()),
            Self::AccpetOr(arg0, arg1) => Self::AccpetOr(arg0.clone(), arg1.clone()),
        }
    }
}

pub struct Lex<'a, M, D>
where
    M: std::fmt::Debug,
    D: DFA<M>,
{
    dfa: &'a D,
    input: &'a str,
    start_pos: usize,
    has_errored: bool,
    phantom_data: PhantomData<M>,
}

impl<'a, M: std::fmt::Debug, D: DFA<M>> Lex<'a, M, D> {
    pub fn new(dfa: &'a D, input: &'a str, start_pos: usize, has_errored: bool) -> Self {
        Self {
            dfa,
            input,
            start_pos,
            has_errored,
            phantom_data: PhantomData,
        }
    }
}

impl<'a, M: std::fmt::Debug, D: DFA<M>> Iterator for Lex<'a, M, D> {
    type Item = anyhow::Result<M>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.has_errored || self.start_pos >= self.input.len() {
            return None;
        }

        // TODO: I don't love that it doesn't error if it fails to lexj
        let (result, new_start) = match self.dfa.get_next_lex(&self.input[self.start_pos..]) {
            Ok(x) => x,
            Err(_) => {
                self.has_errored = true;
                return Some(Err(anyhow::anyhow!("Failed to lex the next token")));
            }
        };
        self.start_pos += new_start;
        Some(Ok(result))
    }
}

#[derive(Debug)]
pub struct DFAStatic<const S: usize, const I: usize, M: std::fmt::Debug> {
    pub d_trans: [[TransitionType<M>; I]; S],
}

impl<const S: usize, const I: usize, M: std::fmt::Debug> std::ops::Index<(usize, char)>
    for DFAStatic<S, I, M>
{
    type Output = TransitionType<M>;

    fn index(&self, (i, a): (usize, char)) -> &Self::Output {
        &self.d_trans[i][a as usize]
    }
}

impl<const S: usize, const I: usize, M: std::fmt::Debug> std::ops::Index<usize>
    for DFAStatic<S, I, M>
{
    type Output = [TransitionType<M>];

    fn index(&self, index: usize) -> &Self::Output {
        &self.d_trans[index]
    }
}

impl<const S: usize, const I: usize, M: std::fmt::Debug> std::ops::IndexMut<(usize, char)>
    for DFAStatic<S, I, M>
{
    fn index_mut(&mut self, (i, a): (usize, char)) -> &mut Self::Output {
        &mut self.d_trans[i][a as usize]
    }
}

impl<const S: usize, const I: usize, M: std::fmt::Debug> std::ops::IndexMut<usize>
    for DFAStatic<S, I, M>
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.d_trans[index]
    }
}

pub trait DFA<M>
where
    M: std::fmt::Debug,
    Self: Sized,
{
    fn states_len(&self) -> usize;
    fn debug_print(&self, letters: &str);
    fn get_next_lex(&self, input: &str) -> anyhow::Result<(M, usize)>;
    fn lex<'a>(&'a self, input: &'a str) -> Lex<'a, M, Self> {
        Lex::new(self, input, 0, false)
    }
    fn is_match(&self, input: &str) -> bool;
    fn contains(&self, input: &str) -> bool;
}

impl<const S: usize, const I: usize, M: std::fmt::Debug> DFA<M> for DFAStatic<S, I, M> {
    fn states_len(&self) -> usize {
        self.d_trans.len()
    }

    fn debug_print(&self, letters: &str) {
        eprintln!("dfa states {:?}\n", self.states_len());
        let letters: std::collections::BTreeSet<_> = letters.chars().collect();
        for i in 0..self.states_len() {
            for a in letters.iter() {
                eprintln!("delta[({}, '{}')] = {:?}", i, a, self[(i, *a)]);
            }
            eprint!("\n");
        }
    }

    fn get_next_lex(&self, input: &str) -> anyhow::Result<(M, usize)> {
        enum ResultState<M> {
            Fail,
            AcceptAt(usize, ConversionFn<M>),
        }

        impl<M> std::fmt::Debug for ResultState<M> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    Self::Fail => write!(f, "Fail"),
                    Self::AcceptAt(arg0, _) => f.debug_tuple("AcceptAt").field(arg0).finish(),
                }
            }
        }

        impl<M> ResultState<M> {
            fn upgrade_at_idx(&mut self, new_val: usize, f: ConversionFn<M>) {
                *self = ResultState::AcceptAt(new_val, f);
            }
        }

        use TransitionType::*;
        let mut state = 0;
        let mut result = ResultState::Fail;

        for (input_idx, a) in input.chars().enumerate() {
            let t = &self[(state, a)];

            match t {
                Normal(i) => {
                    state = *i;
                }
                Fail => {
                    break;
                }
                Accpet(f) => {
                    result = ResultState::AcceptAt(input_idx, f.clone());
                    break;
                }
                AccpetOr(i, f) => {
                    state = *i;
                    result = ResultState::AcceptAt(input_idx, f.clone());
                }
            }
        }

        match result {
            ResultState::AcceptAt(end, f) => f(&input[..end]).map(|x| (x, end)),
            ResultState::Fail => anyhow::bail!("Failed to find a new token"),
        }
    }

    fn is_match(&self, input: &str) -> bool {
        use TransitionType::*;
        let mut state = 0;
        let mut iter = input.chars();

        for a in &mut iter {
            let t = &self[(state, a)];
            match t {
                Normal(i) | AccpetOr(i, _) => state = *i,
                Fail | Accpet(_) => return false,
            }
        }

        self[(state, '\0')].is_accpet()
    }

    fn contains(&self, input: &str) -> bool {
        use TransitionType::*;
        let mut state = 0;

        'outer: for skip in 0..input.len() {
            for a in input.chars().skip(skip) {
                let t = &self[(state, a)];
                match t {
                    Normal(i) => state = *i,
                    Fail => continue 'outer,
                    Accpet(_) | AccpetOr(_, _) => return true,
                }
            }
        }

        return false;
    }
}
