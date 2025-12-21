use std::{
    collections::{BTreeMap, BTreeSet},
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
            Accpet => {}
            AccpetOr(_) => {}
        };
    }
}

#[derive(Debug)]
pub struct DFA {
    d_trans: Vec<Vec<TransitionType>>,
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
        let mut d_trans: Vec<Vec<TransitionType>> = Vec::new();

        let mut i = 0;
        while i < d_states.len() {
            if d_states[i].marked {
                continue;
            }

            eprintln!("states: {:?}", d_states);

            d_states[i].marked = true;

            let mut map: BTreeMap<TerminalNodeElement, BTreeSet<usize>> = BTreeMap::new();
            for (input, node) in refs.iter().enumerate().filter_map(|(j, (_, input))| {
                if !d_states[i].elements.contains(&j) {
                    return None;
                }

                let node: BTreeSet<_> = value.follow_pos[j].clone().into_iter().collect();
                // if node.len() <= 0 {
                //     return None;
                // }

                Some((*input, node))
            }) {
                if let Some(old_set) = map.get_mut(&input) {
                    old_set.extend(node.into_iter())
                } else {
                    map.insert(input, node);
                }
            }

            eprintln!("map of inputs: {:?}", map);

            for (input, state) in map.into_iter() {
                let state_idx = d_states
                    .iter()
                    .enumerate()
                    .find_map(|(i, x)| (x.elements == state).then(|| i))
                    .unwrap_or_else(|| {
                        d_states.push(State::from_set(state.clone()));
                        d_states.len() - 1
                    });

                eprintln!("checking state for existance: {state:?} -> {state_idx:?}");

                if d_trans.len() <= i {
                    // assert that
                    // sizeof usize > sizeof char
                    assert!(usize::MAX > char::MAX as usize);

                    d_trans.extend(vec![
                        vec![
                            TransitionType::Fail;
                            // have an index for all `char as usize`
                            char::MAX as usize + 1
                        ];
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
                        // I think I don't need this
                        // d_trans[i][usize::from(input)] = TransitionType::AccpetOr(state_idx);
                    }
                }
            }

            i += 1;
        }

        Self { d_trans }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_a_or_b_star_paren_abb() {
        let dfa: DFA = "(a|b)*abb".parse().unwrap();
        eprintln!("\nd_trans.len() = {:?}\n", dfa.d_trans.len());

        eprintln!("d_trans[0,a] = {:?}", dfa.d_trans[0]['a' as usize]);
        eprintln!("d_trans[0,b] = {:?}", dfa.d_trans[0]['b' as usize]);
        eprintln!("d_trans[0,c] = {:?}", dfa.d_trans[0]['c' as usize]);

        eprintln!("\nd_trans[1,a] = {:?}", dfa.d_trans[1]['a' as usize]);
        eprintln!("d_trans[1,b] = {:?}", dfa.d_trans[1]['b' as usize]);
        eprintln!("d_trans[1,c] = {:?}", dfa.d_trans[1]['c' as usize]);

        eprintln!("\nd_trans[2,a] = {:?}", dfa.d_trans[2]['a' as usize]);
        eprintln!("d_trans[2,b] = {:?}", dfa.d_trans[2]['b' as usize]);
        eprintln!("d_trans[2,c] = {:?}", dfa.d_trans[2]['c' as usize]);

        eprintln!("\nd_trans[3,a] = {:?}", dfa.d_trans[3]['a' as usize]);
        eprintln!("d_trans[3,b] = {:?}", dfa.d_trans[3]['b' as usize]);
        eprintln!("d_trans[3,c] = {:?}", dfa.d_trans[3]['c' as usize]);
    }
}
