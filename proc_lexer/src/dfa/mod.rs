use std::{
    collections::{BTreeMap, BTreeSet},
    ops::{Index, IndexMut},
    str::FromStr,
};

use crate::trie::{TerminalNodeElement, Trie};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TransitionType {
    Normal(usize),
    Fail,
    Accpet,
    AccpetOr(usize),
}

impl TransitionType {
    fn upgrade(&mut self) {
        use TransitionType::*;

        match self {
            Normal(x) => *self = AccpetOr(*x),
            Fail => *self = Accpet,
            _ => {}
        };
    }

    fn is_accpet(&self) -> bool {
        match self {
            TransitionType::Normal(_) | TransitionType::Fail => false,
            TransitionType::Accpet | TransitionType::AccpetOr(_) => true,
        }
    }
}

const DFA_SIZE: usize = char::MAX as usize + 1;

#[derive(Debug)]
pub struct DFA {
    d_trans: Box<[Box<[TransitionType]>]>,
}

impl Index<(usize, char)> for DFA {
    type Output = TransitionType;

    fn index(&self, (i, a): (usize, char)) -> &Self::Output {
        &self.d_trans[i][a as usize]
    }
}

impl Index<usize> for DFA {
    type Output = [TransitionType];

    fn index(&self, index: usize) -> &Self::Output {
        &self.d_trans[index]
    }
}

impl IndexMut<(usize, char)> for DFA {
    fn index_mut(&mut self, (i, a): (usize, char)) -> &mut Self::Output {
        &mut self.d_trans[i][a as usize]
    }
}

impl IndexMut<usize> for DFA {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.d_trans[index]
    }
}

impl DFA {
    fn states_len(&self) -> usize {
        self.d_trans.len()
    }

    fn debug_print(&self, letters: &str) {
        eprintln!("dfa states {:?}\n", self.states_len());
        let letters: BTreeSet<_> = letters.chars().collect();
        for i in 0..self.states_len() {
            for a in letters.iter() {
                eprintln!("delta[({}, '{}')] = {:?}", i, a, self[(i, *a)]);
            }
            eprint!("\n");
        }
    }

    pub fn is_match(&self, input: &str) -> bool {
        use TransitionType::*;
        let mut state = 0;
        let mut iter = input.chars();

        for a in &mut iter {
            let t = &self[(state, a)];
            // eprintln!("state = {state} a = {a:?}, val = {:?}", t);
            match t {
                Normal(i) | AccpetOr(i) => state = *i,
                Fail | Accpet => return false,
            }
        }

        // eprintln!("state = {state}, val = {:?}", &self[(state, '\0')]);

        self[(state, '\0')].is_accpet()
    }

    pub fn contains(&self, input: &str) -> bool {
        use TransitionType::*;
        let mut state = 0;

        'outer: for skip in 0..input.len() {
            for a in input.chars().skip(skip) {
                let t = &self[(state, a)];
                match t {
                    Normal(i) => state = *i,
                    Fail => continue 'outer,
                    Accpet | AccpetOr(_) => return true,
                }
            }
        }

        return false;
    }
}

impl FromStr for DFA {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let t: Trie = s.parse()?;
        Ok(t.into())
    }
}

impl From<Trie> for DFA {
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

                Some((*input, node))
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
                        d_trans[i][usize::from(input)] = TransitionType::Normal(state_idx)
                    }
                    // TODO: make sure this works
                    TerminalNodeElement::Epsilon => {
                        d_trans[i].iter_mut().for_each(|x| {
                            assert!(*x == TransitionType::Fail);
                            *x = TransitionType::Normal(state_idx);
                        });
                    }
                    TerminalNodeElement::Accept => {
                        d_trans[i].iter_mut().for_each(|x| x.upgrade());
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
    fn test_combine() {
        let combined: DFA = "(if)|(elif)".parse().unwrap();
        combined.debug_print("ifel");

        assert!(combined.is_match("elif"));
        assert!(combined.is_match("if"));
        assert!(!combined.is_match("else"));
        assert!(!combined.is_match("ifelse"));
    }

    #[test]
    fn test_contains() {
        let dfa: DFA = "(a|b)*abb".parse().unwrap();
        assert!(dfa.contains("aaaaaaabbaaaaaaa"));
        assert!(dfa.contains("cabb"));
        assert!(!dfa.contains("cbba"));
        assert!(dfa.contains("cbbabb"));
    }

    #[test]
    fn test_is_match() {
        let dfa: DFA = "(a|b)*abb".parse().unwrap();
        assert!(dfa.is_match("aaaaaabbbbbbbbbbbbbaaaaaaabb"));
        assert!(!dfa.is_match("aaaaaaabbaaaa"));
        assert!(!dfa.is_match("cabb"));
        assert!(!dfa.is_match("abbc"));
        assert!(dfa.is_match("abb"));
    }

    #[test]
    fn test_a_or_b_star_paren_abb() {
        let dfa: DFA = "(a|b)*abb".parse().unwrap();

        use TransitionType::*;
        assert_eq!(dfa.states_len(), 4);
        assert_eq!(dfa[(0, 'a')], Normal(1));
        assert_eq!(dfa[(0, 'b')], Normal(0));
        assert_eq!(dfa[(0, 'c')], Fail);

        assert_eq!(dfa[(1, 'a')], Normal(1));
        assert_eq!(dfa[(1, 'b')], Normal(2));
        assert_eq!(dfa[(1, 'c')], Fail);

        assert_eq!(dfa[(2, 'a')], Normal(1));
        assert_eq!(dfa[(2, 'b')], Normal(3));
        assert_eq!(dfa[(2, 'c')], Fail);

        assert_eq!(dfa[(3, 'a')], AccpetOr(1));
        assert_eq!(dfa[(3, 'b')], AccpetOr(0));
        assert_eq!(dfa[(3, 'c')], Accpet);
    }
}
