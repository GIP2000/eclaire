use thiserror::Error;

use crate::{
    trie::{TerminalNodeElement, Trie, TrieError, TrieNode},
    utils::{VecMap, VecSet},
    AcceptFunc, Lex, LexerError,
};
use std::{
    collections::HashMap,
    fmt::Debug,
    ops::{Index, IndexMut},
};

#[derive(PartialEq, Clone)]
pub enum TransitionType<A: AcceptFunc> {
    Normal(usize),
    Fail,
    Accpet(A),
    AccpetOr(usize, A),
}

#[derive(Debug, Error)]
pub enum DFABuildError {
    #[error(transparent)]
    TrieError(TrieError),
    #[error("Empty regex iterator")]
    EmptyRegexIter,
}

impl<A: AcceptFunc> std::fmt::Debug for TransitionType<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Normal(arg0) => f.debug_tuple("Normal").field(arg0).finish(),
            Self::Fail => write!(f, "Fail"),
            Self::Accpet(_) => write!(f, "Accept"),
            Self::AccpetOr(arg0, _) => f.debug_tuple("AccpetOr").field(arg0).finish(),
        }
    }
}

impl<A: AcceptFunc> TransitionType<A> {
    pub const fn make_fail() -> Self {
        Self::Fail
    }

    pub fn upgrade(&mut self, f: A) {
        use TransitionType::*;

        match self {
            Normal(x) => *self = AccpetOr(*x, f),
            Fail => *self = Accpet(f),
            Accpet(_) => unreachable!("accpet upgrade conflict"),
            AccpetOr(_, _) => unreachable!("accept upgrade conflict"),
        };
    }

    pub fn add_value(&mut self, value: usize) {
        *self = match self {
            TransitionType::Fail => TransitionType::Normal(value),
            TransitionType::Accpet(f) => TransitionType::AccpetOr(value, f.clone()),
            _ => unreachable!("This should never happen we have a conflict"),
        }
    }

    pub fn is_accpet(&self) -> bool {
        match self {
            TransitionType::Normal(_) | TransitionType::Fail => false,
            TransitionType::Accpet(_) | TransitionType::AccpetOr(_, _) => true,
        }
    }
}

pub const DFA_SIZE: usize = u8::MAX as usize + 1;

#[derive(Debug)]
pub struct DFABoxed<A: AcceptFunc> {
    pub d_trans: Box<[Box<[TransitionType<A>]>]>,
}

impl<A> DFA<A> for DFABoxed<A>
where
    A: AcceptFunc + Debug,
{
    fn states_len(&self) -> usize {
        self.d_trans.len()
    }
}

impl<A: AcceptFunc> Index<(usize, u8)> for DFABoxed<A> {
    type Output = TransitionType<A>;

    fn index(&self, (i, a): (usize, u8)) -> &Self::Output {
        &self.d_trans[i][a as usize]
    }
}

impl<A: AcceptFunc> Index<usize> for DFABoxed<A> {
    type Output = [TransitionType<A>];

    fn index(&self, index: usize) -> &Self::Output {
        &self.d_trans[index]
    }
}

impl<A> IndexMut<(usize, u8)> for DFABoxed<A>
where
    A: AcceptFunc,
{
    fn index_mut(&mut self, (i, a): (usize, u8)) -> &mut Self::Output {
        &mut self.d_trans[i][a as usize]
    }
}

impl<A> IndexMut<usize> for DFABoxed<A>
where
    A: AcceptFunc,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.d_trans[index]
    }
}

impl<A> DFABoxed<A>
where
    A: AcceptFunc + Eq,
{
    pub fn from_regexes<S: AsRef<str>, I: Iterator<Item = (S, A)>>(
        mut iter: I,
    ) -> Result<Self, DFABuildError> {
        let mut size = 0;
        let mut root = if let Some((regex, accept)) = iter.next() {
            TrieNode::build_from_regex(regex.as_ref(), accept.clone(), &mut size, 0)
                .map_err(|err| DFABuildError::TrieError(err))?
        } else {
            return Err(DFABuildError::EmptyRegexIter);
        };

        for (rank, (regex, accept)) in iter.enumerate() {
            root =
                TrieNode::or_from_regex(root, regex.as_ref(), accept.clone(), &mut size, rank + 1)
                    .map_err(|err| DFABuildError::TrieError(err))?;
        }

        let follow_pos = root.calculate_follow_pos(size);

        let trie = Trie {
            root,
            follow_pos,
            size,
        };

        Ok(trie.into())
    }
}

impl<A> From<Trie<A>> for DFABoxed<A>
where
    A: AcceptFunc + Eq,
{
    fn from(value: Trie<A>) -> Self {
        #[derive(Debug)]
        struct State {
            elements: VecSet<usize>,
            marked: bool,
        }

        impl State {
            pub fn from_set(elements: VecSet<usize>) -> Self {
                Self {
                    elements,
                    marked: false,
                }
            }
        }

        let first_elements: VecSet<_> = value
            .root
            .get_meta()
            .first_pos
            .clone()
            .into_iter()
            .collect();

        let refs = value.root.get_refs();
        let mut d_states = vec![State::from_set(first_elements.clone())];
        let mut d_trans: Vec<Box<[TransitionType<A>]>> = Vec::new();

        let mut rank_map: HashMap<(usize, usize), usize> = HashMap::new();

        let mut i = 0;
        while i < d_states.len() {
            if d_states[i].marked {
                continue;
            }

            d_states[i].marked = true;

            let mut map: VecMap<TerminalNodeElement<A>, VecSet<usize>> = VecMap::new();
            for (input, node) in refs.iter().enumerate().filter_map(|(j, (_, input))| {
                if !d_states[i].elements.contains(&j) {
                    return None;
                }

                let node: VecSet<_> = value.follow_pos[j].clone().into_iter().collect();

                Some((input.clone(), node))
            }) {
                if let Some(old_set) = map.get_mut(&input) {
                    old_set.extend(node.into_iter())
                } else {
                    map.insert(input, node);
                }
            }

            for (input, state) in map.into_iter() {
                let state_idx = d_states
                    .iter()
                    .enumerate()
                    .find_map(|(i, x)| (x.elements == state).then(|| i))
                    .unwrap_or_else(|| {
                        d_states.push(State::from_set(state.clone()));
                        d_states.len() - 1
                    });

                if d_trans.len() <= i {
                    d_trans.extend(vec![
                        vec![
                            TransitionType::make_fail();
                            // have an index for all `char as usize`
                            DFA_SIZE
                        ]
                        .into_boxed_slice();
                        i - d_trans.len() + 1
                    ]);
                }

                match input {
                    TerminalNodeElement::Char(_) => {
                        d_trans[i][usize::from(input)].add_value(state_idx);
                    }
                    TerminalNodeElement::Accept(f, current_rank) => {
                        d_trans[i].iter_mut().enumerate().for_each(|(a, x)| {
                            let old_rank = rank_map.get(&(i, a));

                            match old_rank {
                                Some(old) if *old > current_rank => {
                                    // eprintln!(
                                    //     "overwritting: {:?}, old = {:?}, current = {:?}",
                                    //     (i, a),
                                    //     old,
                                    //     current_rank
                                    // );
                                    x.upgrade(f.clone());
                                    rank_map.insert((i, a), current_rank);
                                }
                                None => {
                                    // eprintln!(
                                    //     "not overwritting (None): {:?},  current = {:?}",
                                    //     (i, a),
                                    //     current_rank
                                    // );

                                    x.upgrade(f.clone());
                                    rank_map.insert((i, a), current_rank);
                                }
                                _ => {
                                    // eprintln!(
                                    //     "not overwritting (otherwise): {:?},  current = {:?}",
                                    //     (i, a),
                                    //     current_rank
                                    // );
                                }
                            }
                        });
                    }
                }
            }

            i += 1;
        }

        Self {
            d_trans: d_trans.into_boxed_slice(),
        }
    }
}

#[derive(Debug)]
pub struct DFAStatic<const S: usize, const I: usize, A>
where
    A: AcceptFunc,
{
    pub d_trans: [[TransitionType<A>; I]; S],
}

impl<const S: usize, const I: usize, A> std::ops::Index<(usize, u8)> for DFAStatic<S, I, A>
where
    A: AcceptFunc,
{
    type Output = TransitionType<A>;

    fn index(&self, (i, a): (usize, u8)) -> &Self::Output {
        &self.d_trans[i][a as usize]
    }
}

impl<const S: usize, const I: usize, A> std::ops::Index<usize> for DFAStatic<S, I, A>
where
    A: AcceptFunc,
{
    type Output = [TransitionType<A>];

    fn index(&self, index: usize) -> &Self::Output {
        &self.d_trans[index]
    }
}

impl<const S: usize, const I: usize, A> std::ops::IndexMut<(usize, u8)> for DFAStatic<S, I, A>
where
    A: AcceptFunc,
{
    fn index_mut(&mut self, (i, a): (usize, u8)) -> &mut Self::Output {
        &mut self.d_trans[i][a as usize]
    }
}

impl<const S: usize, const I: usize, A> std::ops::IndexMut<usize> for DFAStatic<S, I, A>
where
    A: AcceptFunc,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.d_trans[index]
    }
}

pub trait DFA<A>
where
    A: AcceptFunc,
    Self: Sized
        + Index<usize, Output = [TransitionType<A>]>
        + Index<(usize, u8), Output = TransitionType<A>>,
    for<'a> A::Output<'a>: std::fmt::Debug,
{
    fn states_len(&self) -> usize;

    fn debug_print2(&self, letters: &str, print_override: impl Fn(&TransitionType<A>) -> String) {
        eprintln!("dfa states {:?}\n", self.states_len());
        let letters: std::collections::BTreeSet<_> = letters.bytes().collect();
        for i in 0..self.states_len() {
            for a in letters.iter() {
                let print = if a.is_ascii_alphanumeric() || *a == b'.' {
                    &(*a as char) as &dyn std::fmt::Debug
                } else {
                    &a as &dyn std::fmt::Debug
                };
                eprintln!(
                    "delta[({}, {:?})] = {:?}",
                    i,
                    print,
                    print_override(&self[(i, *a)])
                );
            }
            eprint!("\n");
        }
    }

    fn debug_print(&self, letters: &str) {
        eprintln!("dfa states {:?}\n", self.states_len());
        let letters: std::collections::BTreeSet<_> = letters.bytes().collect();
        for i in 0..self.states_len() {
            for a in letters.iter() {
                let print = if (b'a'..=b'z').contains(&a) {
                    &(*a as char) as &dyn std::fmt::Debug
                } else {
                    &a as &dyn std::fmt::Debug
                };
                eprintln!("delta[({}, {:?})] = {:?}", i, print, self[(i, *a)]);
            }
            eprint!("\n");
        }
    }

    fn get_next_lex<'a>(
        &self,
        input: &'a str,
    ) -> Result<(A::Output<'a>, usize), LexerError<A::Error>> {
        use TransitionType::*;
        let mut state = 0;
        let mut result = Err(LexerError::MatchNotFound);

        for (input_idx, a) in input.bytes().chain(std::iter::once(b'\0')).enumerate() {
            let t = &self[(state, a)];

            match t {
                Normal(i) => {
                    state = *i;
                }
                Fail => {
                    break;
                }
                Accpet(f) => {
                    result = Ok((input_idx, f.clone()));
                    break;
                }
                AccpetOr(i, f) => {
                    state = *i;
                    result = Ok((input_idx, f.clone()));
                }
            }
        }

        result.and_then(|(end, f)| {
            f.convert(&input[..end])
                .map_err(|err| err.into())
                .map(|x| (x, end))
        })
    }

    fn lex<'d, 'a>(&'d self, input: &'a str) -> Lex<'a, 'd, A, Self> {
        Lex::new(self, input, 0, false)
    }

    fn is_match(&self, input: &str) -> bool {
        use TransitionType::*;
        let mut state = 0;
        let mut iter = input.bytes();

        for a in &mut iter {
            let t = &self[(state, a)];
            match t {
                Normal(i) | AccpetOr(i, _) => state = *i,
                Fail | Accpet(_) => return false,
            }
        }

        self[(state, b'\0')].is_accpet()
    }

    fn contains(&self, input: &str) -> bool {
        use TransitionType::*;
        let mut state = 0;

        'outer: for skip in 0..input.len() {
            for a in input.bytes().skip(skip) {
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

impl<const S: usize, const I: usize, A> DFA<A> for DFAStatic<S, I, A>
where
    A: AcceptFunc,
{
    fn states_len(&self) -> usize {
        self.d_trans.len()
    }
}

#[cfg(test)]
mod test {
    use crate::LexerOutput;
    use std::error::Error;

    type BError = Box<dyn Error>;

    type Result<T> = std::result::Result<T, BError>;

    use super::*;

    fn foo<'a>(x: &'a str) -> Result<&'a str> {
        Ok(x.trim())
    }

    fn bar<'a>(x: &'a str) -> Result<&'a str> {
        Ok(x)
    }

    type F2 = for<'a> fn(&'a str) -> Result<char>;

    fn a<'a>(_x: &'a str) -> Result<char> {
        Ok('a')
    }

    fn b<'a>(_x: &'a str) -> Result<char> {
        Ok('b')
    }

    fn s<'a>(_x: &'a str) -> Result<char> {
        Ok(' ')
    }

    impl AcceptFunc for F2 {
        type Error = Box<dyn Error>;
        type Output<'a> = char;

        fn convert<'a>(&self, input: &'a str) -> Result<Self::Output<'a>> {
            self(input)
        }
    }

    type F = for<'a> fn(&'a str) -> Result<&'a str>;

    impl AcceptFunc for F {
        type Error = Box<dyn Error>;
        type Output<'a> = &'a str;

        fn convert<'a>(&self, input: &'a str) -> Result<Self::Output<'a>> {
            self(input)
        }
    }

    #[test]
    fn test_char() {
        let dfa: DFABoxed<_> = Trie::from_regex("'[^\n' ]'", bar as F).unwrap().into();

        dfa.debug_print("'\n ^[]abc");
        assert!(dfa.is_match("'a'"));
    }

    #[test]
    fn test_ident_plus_fn() {
        #[derive(Debug, Eq, PartialEq, PartialOrd, Ord, Clone)]
        enum Tokens {
            Fn,
            Ident,
            Skip,
        }

        type F = for<'a> fn(&'a str) -> Result<Tokens>;

        #[derive(Debug, Clone, Hash, Eq)]
        struct FnContainer(F, &'static str);

        impl PartialEq for FnContainer {
            fn eq(&self, other: &Self) -> bool {
                self.1 == other.1
            }
        }

        impl AcceptFunc for FnContainer {
            type Error = BError;
            type Output<'a> = Tokens;

            fn convert<'a>(&self, input: &'a str) -> Result<Self::Output<'a>> {
                self.0(input)
            }
        }

        use Tokens::*;

        fn parse_fn<'a>(_: &'a str) -> Result<Tokens> {
            Ok(Fn)
        }
        fn parse_ident(_: &str) -> Result<Tokens> {
            Ok(Ident)
        }
        fn parse_skip(_: &str) -> Result<Tokens> {
            Ok(Skip)
        }

        let x: [(&str, FnContainer); 3] = [
            ("fn", FnContainer(parse_fn, "fn")),
            ("[a-zA-Z][a-zA-Z0-9_]*", FnContainer(parse_ident, "ident")),
            ("[ \n]", FnContainer(parse_skip, "skip")),
        ];
        let dfa: DFABoxed<_> = DFABoxed::from_regexes(x.into_iter()).unwrap();

        dfa.debug_print2("fn fnap  \0", |x| match x {
            TransitionType::Accpet(y) => format!("Accept({:?})", y.1),
            TransitionType::AccpetOr(x, y) => format!("AcceptOr({:?}, {:?})", x, y.1),
            x => format!("{:?}", x),
        });

        // let mut lex = dfa.lex("fn clap fn").filter_map(|x| match x {
        //     Ok((Skip, _)) => None,
        //     Ok((x, _)) => Some(Ok(x)),
        //     Err(er) => Some(Err(er)),
        // });

        let mut lex = dfa.lex("fn clap fn").filter_map(|x| match x {
            Ok(LexerOutput {
                meta: _,
                data: Skip,
            }) => None,
            Ok(LexerOutput { meta: _, data: x }) => Some(Ok(x)),
            Err(er) => Some(Err(er)),
        });

        assert_eq!(lex.next().unwrap().unwrap(), Fn);
        assert_eq!(lex.next().unwrap().unwrap(), Ident);
        assert_eq!(lex.next().unwrap().unwrap(), Fn);
        assert!(lex.next().is_none());
    }

    #[test]
    fn test_float() {
        let trie = Trie::from_regex("[0-3][0-3]*\\.[0-3]*", "float").unwrap();

        let dfa: DFABoxed<_> = trie.into();

        assert!(dfa.is_match("1."));
        assert!(dfa.is_match("12."));
        assert!(dfa.is_match("12.2"));
        assert!(dfa.is_match("12.23"));
        assert!(!dfa.is_match("1"));
        assert!(!dfa.is_match("1233"));
        assert!(!dfa.is_match(".1233"));
    }

    #[test]
    fn test_float_v_int() {
        let x: [(Box<str>, F2); 3] = [
            ("[0-9][0-9]*\\.[0-9]*".into(), b),
            ("[0-9][0-9]*".into(), a),
            (" ".into(), s),
        ];

        let combined: DFABoxed<_> = DFABoxed::from_regexes(x.into_iter()).unwrap();

        combined.debug_print2("0123456789.a\0", |x| match x {
            TransitionType::Accpet(y) => format!("Accept({:?})", y("")),
            TransitionType::AccpetOr(x, y) => format!("AcceptOr({:?}, {:?})", x, y("")),
            x => format!("{:?}", x),
        });

        let input = "111 111.1 111.";

        let mut lexer = combined.lex(input).filter_map(|x| {
            x.ok()
                .and_then(|LexerOutput { meta: _, data }| (data != ' ').then_some(data))
        });

        assert_eq!(lexer.next().unwrap(), 'a');
        assert_eq!(lexer.next().unwrap(), 'b');
        assert_eq!(lexer.next().unwrap(), 'b');
        assert!(lexer.next().is_none());
    }

    #[test]
    fn test_lex() {
        let x: [(Box<str>, F); 4] = [
            ("if".into(), bar),
            ("else".into(), bar),
            (" ".into(), foo),
            (
                "\"(a|b|c|d|e|f|g|h|i|j|k|l|m|n|o|p|q|r|s|t|u|v|w|x|y|z| )*\"".into(),
                bar,
            ),
        ];

        let combined: DFABoxed<_> = DFABoxed::from_regexes(x.into_iter()).unwrap();
        let input = "if else \"hi there my name is greg\" if else   \"this is really crazy\" if";

        let mut lex_iter = combined.lex(input).filter_map(|x| {
            x.ok()
                .and_then(|LexerOutput { meta: _, data: x }| (!x.is_empty()).then_some(x))
        });

        assert_eq!(lex_iter.next().unwrap(), "if");
        assert_eq!(lex_iter.next().unwrap(), "else");
        assert_eq!(lex_iter.next().unwrap(), "\"hi there my name is greg\"");
        assert_eq!(lex_iter.next().unwrap(), "if");
        assert_eq!(lex_iter.next().unwrap(), "else");
        assert_eq!(lex_iter.next().unwrap(), "\"this is really crazy\"");
        assert_eq!(lex_iter.next().unwrap(), "if");
        assert_eq!(lex_iter.next(), None);
    }

    #[test]
    fn test_paren() {
        let dfa: DFABoxed<_> = Trie::from_regex("da*|b", "to_uppercase").unwrap().into();

        assert!(dfa.is_match("b"));
        assert!(dfa.is_match("d"));
        assert!(dfa.is_match("daaa"));
        assert!(!dfa.is_match("daaab"));
    }

    #[test]
    fn test_string_type() {
        let trie = Trie::from_regex("c(a|b)*c", "to_uppercase").unwrap();
        let dfa: DFABoxed<_> = trie.into();
        dfa.debug_print("cab\0d");
        //
        assert!(dfa.is_match("caaaabbbbc"));
    }

    #[test]
    fn test_grab_match() {
        let dfa: DFABoxed<_> = Trie::from_regex("(a|b)*aab", "to_uppercase")
            .unwrap()
            .into();
        let input = "aaaaaabb";

        let result = dfa.get_next_lex(input).unwrap();
        assert_eq!(result.0, "aaaaaab");
        assert_eq!(result.1, input.len() - 1);

        assert!(matches!(dfa.get_next_lex("caaab"), Err(_)))
    }

    #[test]
    fn test_combine_nested() {
        let x: [(Box<str>, _); 3] = [("==".into(), ""), ("=".into(), ""), (" ".into(), "")];

        let combined: DFABoxed<_> = DFABoxed::from_regexes(x.into_iter()).unwrap();
        let input = "= == === = =";

        let mut lex_iter = combined.lex(input).filter_map(|x| {
            x.ok()
                .and_then(|LexerOutput { meta: _, data: x }| (x != " ").then_some(x))
        });

        assert_eq!(lex_iter.next().unwrap(), "=");
        assert_eq!(lex_iter.next().unwrap(), "==");
        assert_eq!(lex_iter.next().unwrap(), "==");
        assert_eq!(lex_iter.next().unwrap(), "=");
        assert_eq!(lex_iter.next().unwrap(), "=");
        assert_eq!(lex_iter.next().unwrap(), "=");
        assert_eq!(lex_iter.next(), None);
    }

    #[test]
    fn test_combine() {
        let x: [(_, _); 2] = [("if", ""), ("elif", "")];
        let combined: DFABoxed<_> = DFABoxed::from_regexes(x.into_iter()).unwrap();

        assert!(combined.is_match("elif"));
        assert!(combined.is_match("if"));
        assert!(!combined.is_match("else"));
        assert!(!combined.is_match("ifelse"));
    }

    #[test]
    fn test_contains() {
        let dfa: DFABoxed<_> = Trie::from_regex("(a|b)*abb", "").unwrap().into();

        assert!(dfa.contains("aaaaaaabbaaaaaaa"));
        assert!(dfa.contains("cabb"));
        assert!(!dfa.contains("cbba"));
        assert!(dfa.contains("cbbabb"));
    }

    #[test]
    fn test_is_match() {
        let dfa: DFABoxed<_> = Trie::from_regex("(a|b)*abb", "").unwrap().into();
        assert!(dfa.is_match("aaaaaabbbbbbbbbbbbbaaaaaaabb"));
        assert!(!dfa.is_match("aaaaaaabbaaaa"));
        assert!(!dfa.is_match("cabb"));
        assert!(!dfa.is_match("abbc"));
        assert!(dfa.is_match("abb"));
    }

    #[test]
    fn test_ident() {
        let dfa: DFABoxed<_> = Trie::from_regex("[a-zA-Z_][a-zA-Z0-9_]*", "")
            .unwrap()
            .into();

        assert!(dfa.is_match("abbc_123"));
        assert!(!dfa.is_match("abb c_123"));
        assert!(!dfa.is_match("1abbc_123"));
        assert!(dfa.is_match("_abbc_123"));
    }

    #[test]
    fn test_dot() {
        let dfa: DFABoxed<_> = Trie::from_regex("\".*\"", "").unwrap().into();
        assert!(dfa.is_match("\"hello there my name is aASDF!@#A12309adsjfklasdf'\\DF\""));
        assert!(!dfa.is_match("\"hello there my name is aASDF!@#ADF"));
        assert!(!dfa.is_match("hello there my name is aASDF!@#ADF"));
        assert!(!dfa.is_match("hello there my name is aASDF!@#ADF\""));
    }

    #[test]
    fn test_a_or_b_star_paren_abb() {
        let dfa: DFABoxed<_> = Trie::from_regex("(a|b)*abb", "").unwrap().into();

        use TransitionType::*;

        // dfa.debug_print("abc\0");

        assert_eq!(dfa.states_len(), 4);
        assert_eq!(dfa[(0, b'a')], Normal(1));
        assert_eq!(dfa[(0, b'b')], Normal(0));
        assert_eq!(dfa[(0, b'c')], TransitionType::make_fail());

        assert_eq!(dfa[(1, b'a')], Normal(1));
        assert_eq!(dfa[(1, b'b')], Normal(2));
        assert_eq!(dfa[(1, b'c')], TransitionType::make_fail());

        assert_eq!(dfa[(2, b'a')], Normal(1));
        assert_eq!(dfa[(2, b'b')], Normal(3));
        assert_eq!(dfa[(2, b'c')], TransitionType::make_fail());

        // assert_eq!(dfa[(3, 'a')], AccpetOr(1));
        assert!(
            if let AccpetOr(1, _) = dfa[(3, b'a')] {
                true
            } else {
                false
            },
            "Failed expected AcceptOr(1) found: {:?}",
            dfa[(3, b'a')]
        );
        // assert_eq!(dfa[(3, 'b')], AccpetOr(0));
        assert!(
            if let AccpetOr(0, _) = dfa[(3, b'b')] {
                true
            } else {
                false
            },
            "Failed expected AcceptOr(0) found: {:?}",
            dfa[(3, b'b')]
        );

        // assert_eq!(dfa[(3, 'c')], Accpet);
        assert!(
            if let Accpet(_) = dfa[(3, b'c')] {
                true
            } else {
                false
            },
            "Failed expected Accept found: {:?}",
            dfa[(3, b'c')]
        );
    }
}
