use anyhow::bail;
use lexer::DFA;
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Debug,
    ops::{Index, IndexMut},
};

use crate::trie::{TerminalNodeElement, Trie, TrieNode};

#[derive(PartialEq)]
pub enum TransitionType {
    Normal(usize),
    Fail,
    Accpet(Box<str>),
    AccpetOr(usize, Box<str>),
}

impl std::fmt::Debug for TransitionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Normal(arg0) => f.debug_tuple("Normal").field(arg0).finish(),
            Self::Fail => write!(f, "Fail"),
            Self::Accpet(_) => write!(f, "Accept"),
            Self::AccpetOr(arg0, _) => f.debug_tuple("AccpetOr").field(arg0).finish(),
        }
    }
}

impl TransitionType {
    pub fn upgrade(&mut self, f: Box<str>) {
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

impl Clone for TransitionType {
    fn clone(&self) -> Self {
        match self {
            Self::Normal(arg0) => Self::Normal(arg0.clone()),
            Self::Fail => Self::Fail,
            Self::Accpet(arg0) => Self::Accpet(arg0.clone()),
            Self::AccpetOr(arg0, arg1) => Self::AccpetOr(arg0.clone(), arg1.clone()),
        }
    }
}

pub(crate) const DFA_SIZE: usize = u8::MAX as usize + 1;

#[derive(Debug)]
pub(crate) struct DFABoxed {
    pub(crate) d_trans: Box<[Box<[TransitionType]>]>,
}

impl<'a> DFA<'a, Box<str>> for DFABoxed {
    fn states_len(&self) -> usize {
        self.d_trans.len()
    }

    fn debug_print(&self, letters: &str) {
        eprintln!("dfa states {:?}\n", self.states_len());
        let letters: BTreeSet<_> = letters.bytes().collect();
        for i in 0..self.states_len() {
            for a in letters.iter() {
                eprintln!("delta[({}, '{}')] = {:?}", i, a, self[(i, *a)]);
            }
            eprint!("\n");
        }
    }

    fn get_next_lex(&self, input: &str) -> anyhow::Result<(Box<str>, usize)> {
        enum ResultState {
            Fail,
            AcceptAt(usize, Box<str>),
        }

        impl Debug for ResultState {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    Self::Fail => write!(f, "Fail"),
                    Self::AcceptAt(arg0, _) => f.debug_tuple("AcceptAt").field(arg0).finish(),
                }
            }
        }

        impl ResultState {
            fn upgrade_at_idx(&mut self, new_val: usize, f: Box<str>) {
                *self = ResultState::AcceptAt(new_val, f);
            }
        }

        use TransitionType::*;
        let mut state = 0;
        let mut result = ResultState::Fail;

        for (input_idx, a) in input.bytes().enumerate() {
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
            ResultState::AcceptAt(end, _) => Ok((input[..end].into(), end)),
            ResultState::Fail => bail!("Failed to find a new token"),
        }
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

impl Index<(usize, u8)> for DFABoxed {
    type Output = TransitionType;

    fn index(&self, (i, a): (usize, u8)) -> &Self::Output {
        &self.d_trans[i][a as usize]
    }
}

impl Index<usize> for DFABoxed {
    type Output = [TransitionType];

    fn index(&self, index: usize) -> &Self::Output {
        &self.d_trans[index]
    }
}

impl IndexMut<(usize, u8)> for DFABoxed {
    fn index_mut(&mut self, (i, a): (usize, u8)) -> &mut Self::Output {
        &mut self.d_trans[i][a as usize]
    }
}

impl IndexMut<usize> for DFABoxed {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.d_trans[index]
    }
}

impl DFABoxed {
    pub fn from_regexes<I: Iterator<Item = (Box<str>, Box<str>)>>(
        mut iter: I,
    ) -> anyhow::Result<Self> {
        let mut size = 0;
        let mut root = if let Some((regex, accept)) = iter.next() {
            TrieNode::build_from_regex(&regex, accept.clone(), &mut size)?
        } else {
            anyhow::bail!("Empty array found");
        };

        for (regex, accept) in iter {
            root = TrieNode::or_from_regex(root, &regex, accept.clone(), &mut size)?;
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

impl From<Trie> for DFABoxed {
    fn from(value: Trie) -> Self {
        #[derive(Debug)]
        struct State {
            elements: BTreeSet<usize>,
            marked: bool,
        }

        impl State {
            pub fn from_set(elements: BTreeSet<usize>) -> Self {
                Self {
                    elements,
                    marked: false,
                }
            }
        }

        let first_elements: BTreeSet<_> = value
            .root
            .get_meta()
            .first_pos
            .clone()
            .into_iter()
            .collect();

        let refs = value.root.get_refs();
        let mut d_states = vec![State::from_set(first_elements.clone())];
        let mut d_trans: Vec<Box<[TransitionType]>> = Vec::new();

        let mut i = 0;
        while i < d_states.len() {
            if d_states[i].marked {
                continue;
            }

            d_states[i].marked = true;

            let mut map: BTreeMap<TerminalNodeElement, BTreeSet<usize>> = BTreeMap::new();
            for (input, node) in refs.iter().enumerate().filter_map(|(j, (_, input))| {
                if !d_states[i].elements.contains(&j) {
                    return None;
                }

                let node: BTreeSet<_> = value.follow_pos[j].clone().into_iter().collect();

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
                    // assert that
                    // sizeof usize > sizeof char
                    assert!(usize::MAX > char::MAX as usize);

                    d_trans.extend(vec![
                        vec![
                            TransitionType::Fail;
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
                    // TODO: make sure this works
                    TerminalNodeElement::Epsilon => {
                        unimplemented!("Need to rethink this")
                    }
                    TerminalNodeElement::Accept(f) => {
                        d_trans[i].iter_mut().for_each(|x| x.upgrade(f.clone()));
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

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_lex() {
        let x: [(Box<str>, Box<str>); 4] = [
            ("if".into(), "".into()),
            ("else".into(), "".into()),
            (" ".into(), "".into()),
            (
                "\"(a|b|c|d|e|f|g|h|i|j|k|l|m|n|o|p|q|r|s|t|u|v|w|x|y|z| )*\"".into(),
                "".into(),
            ),
        ];

        let combined: DFABoxed = DFABoxed::from_regexes(x.into_iter()).unwrap();
        let input = "if else \"hi there my name is greg\" if else   \"this is really crazy\" if";

        let mut lex_iter = combined
            .lex(input)
            .filter_map(|x| x.ok().and_then(|x| (x != " ".into()).then_some(x)));

        assert_eq!(lex_iter.next().unwrap(), "if".into());
        assert_eq!(lex_iter.next().unwrap(), "else".into());
        assert_eq!(
            lex_iter.next().unwrap(),
            "\"hi there my name is greg\"".into()
        );
        assert_eq!(lex_iter.next().unwrap(), "if".into());
        assert_eq!(lex_iter.next().unwrap(), "else".into());
        assert_eq!(lex_iter.next().unwrap(), "\"this is really crazy\"".into());
        assert_eq!(lex_iter.next(), None);
    }

    #[test]
    fn test_paren() {
        let dfa: DFABoxed = Trie::from_regex("da*|b", "to_uppercase".into())
            .unwrap()
            .into();

        assert!(dfa.is_match("b"));
        assert!(dfa.is_match("d"));
        assert!(dfa.is_match("daaa"));
        assert!(!dfa.is_match("daaab"));
    }

    #[test]
    fn test_string_type() {
        let trie = Trie::from_regex("c(a|b)*c", "to_uppercase".into()).unwrap();
        eprintln!("trie: {:?}", trie);
        let dfa: DFABoxed = trie.into();
        dfa.debug_print("cab\0d");
        //
        assert!(dfa.is_match("caaaabbbbc"));
    }

    #[test]
    fn test_grab_match() {
        let dfa: DFABoxed = Trie::from_regex("(a|b)*aab", "to_uppercase".into())
            .unwrap()
            .into();
        let input = "aaaaaabb";

        let result = dfa.get_next_lex(input).unwrap();
        assert_eq!(result.0, "aaaaaab".into());
        assert_eq!(result.1, input.len() - 1);

        assert!(matches!(dfa.get_next_lex("caaab"), Err(_)))
    }

    #[test]
    fn test_combine() {
        let x: [(Box<str>, Box<str>); 2] = [("if".into(), "".into()), ("elif".into(), "".into())];
        let combined: DFABoxed = DFABoxed::from_regexes(x.into_iter()).unwrap();

        assert!(combined.is_match("elif"));
        assert!(combined.is_match("if"));
        assert!(!combined.is_match("else"));
        assert!(!combined.is_match("ifelse"));
    }

    #[test]
    fn test_contains() {
        let dfa: DFABoxed = Trie::from_regex("(a|b)*abb", "".into()).unwrap().into();

        assert!(dfa.contains("aaaaaaabbaaaaaaa"));
        assert!(dfa.contains("cabb"));
        assert!(!dfa.contains("cbba"));
        assert!(dfa.contains("cbbabb"));
    }

    #[test]
    fn test_is_match() {
        let dfa: DFABoxed = Trie::from_regex("(a|b)*abb", "".into()).unwrap().into();
        assert!(dfa.is_match("aaaaaabbbbbbbbbbbbbaaaaaaabb"));
        assert!(!dfa.is_match("aaaaaaabbaaaa"));
        assert!(!dfa.is_match("cabb"));
        assert!(!dfa.is_match("abbc"));
        assert!(dfa.is_match("abb"));
    }

    #[test]
    fn test_a_or_b_star_paren_abb() {
        let dfa: DFABoxed = Trie::from_regex("(a|b)*abb", "".into()).unwrap().into();

        use TransitionType::*;

        // dfa.debug_print("abc\0");

        assert_eq!(dfa.states_len(), 4);
        assert_eq!(dfa[(0, b'a')], Normal(1));
        assert_eq!(dfa[(0, b'b')], Normal(0));
        assert_eq!(dfa[(0, b'c')], Fail);

        assert_eq!(dfa[(1, b'a')], Normal(1));
        assert_eq!(dfa[(1, b'b')], Normal(2));
        assert_eq!(dfa[(1, b'c')], Fail);

        assert_eq!(dfa[(2, b'a')], Normal(1));
        assert_eq!(dfa[(2, b'b')], Normal(3));
        assert_eq!(dfa[(2, b'c')], Fail);

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
