use anyhow::{anyhow, bail, Result};
use std::{
    collections::{HashSet, VecDeque},
    iter::Peekable,
};

#[derive(Debug, Default, PartialEq, Clone)]
pub(crate) struct TrieMeta {
    pub(crate) nullable: bool,
    pub(crate) first_pos: HashSet<usize>,
    pub(crate) last_pos: HashSet<usize>,
}

impl TrieMeta {
    fn calculate_first_pass_for_cat(l: &Self, r: &Self) -> Self {
        let first_pos = if l.nullable {
            l.first_pos.union(&r.first_pos).cloned().collect()
        } else {
            l.first_pos.clone()
        };

        let last_pos = if r.nullable {
            l.first_pos.union(&r.first_pos).cloned().collect()
        } else {
            r.first_pos.clone()
        };

        Self {
            first_pos,
            last_pos,
            nullable: r.nullable && l.nullable,
        }
    }

    fn calculate_first_pass_for_or(l: &Self, r: &Self) -> Self {
        Self {
            nullable: r.nullable || l.nullable,
            first_pos: r.first_pos.union(&l.first_pos).cloned().collect(),
            last_pos: r.last_pos.union(&l.last_pos).cloned().collect(),
        }
    }

    fn calculate_first_pass_for_star(x: &Self) -> Self {
        Self {
            nullable: true,
            ..x.clone()
        }
    }

    fn calculate_first_pass_from_char<M>(c: &TerminalNodeElement<M>, index: usize) -> Self {
        let set = HashSet::from([index]);
        Self {
            nullable: c.is_nullable(),
            first_pos: set.clone(),
            last_pos: set,
        }
    }
}

#[derive(Debug, PartialEq)]
pub(crate) struct Trie<M> {
    pub(crate) root: TrieNode<M>,
    pub(crate) follow_pos: Vec<HashSet<usize>>,
    pub(crate) size: usize,
}

impl<M: std::fmt::Debug> Trie<M> {
    pub(crate) fn from_regex(regex: &str, accept: fn(&str) -> Result<M>) -> Result<Self> {
        let mut size = 0;

        let root: TrieNode<M> =
            TrieNode::from_iterator(&mut regex.chars().map(|x| x.into()).peekable(), &mut size)?
                // Add the accept state to the end
                .cat(TrieNode::<M>::terminal(
                    TerminalNodeElement::Accept(accept),
                    size,
                ));
        // increment the size to accomidate the accept state
        size += 1;

        let follow_pos = root.calculate_follow_pos(size);

        Ok(Self {
            root,
            follow_pos,
            size,
        })
    }
}

pub(crate) type ConversionFn<M> = fn(&str) -> Result<M>;

#[derive(Debug, Eq, Hash)]
pub(crate) enum TerminalNodeElement<M> {
    Char(char),
    Epsilon,
    Accept(ConversionFn<M>),
}

impl<M> Copy for TerminalNodeElement<M> {}

impl<M> Clone for TerminalNodeElement<M> {
    fn clone(&self) -> Self {
        match self {
            Self::Char(arg0) => Self::Char(arg0.clone()),
            Self::Epsilon => Self::Epsilon,
            Self::Accept(arg0) => Self::Accept(arg0.clone()),
        }
    }
}

impl<M: Eq> Ord for TerminalNodeElement<M> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl<M> PartialOrd for TerminalNodeElement<M> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering;
        use TerminalNodeElement::*;
        match (self, other) {
            (Char(a), Char(b)) => a.partial_cmp(b),
            (Char(_), _) | (Accept(_), Epsilon) => Some(Ordering::Greater),
            (Accept(_), Char(_)) | (Epsilon, Char(_)) | (Epsilon, Accept(_)) => {
                Some(Ordering::Less)
            }
            (Accept(_), Accept(_)) | (Epsilon, Epsilon) => Some(Ordering::Equal),
        }
    }
}

impl<M> PartialEq for TerminalNodeElement<M> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Char(l0), Self::Char(r0)) => l0 == r0,
            _ => core::mem::discriminant(self) == core::mem::discriminant(other),
        }
    }
}

impl<M> From<TerminalNodeElement<M>> for usize {
    fn from(value: TerminalNodeElement<M>) -> Self {
        match value {
            TerminalNodeElement::Char(x) => x as usize,
            TerminalNodeElement::Epsilon => unimplemented!("Think through epsilon more"),
            TerminalNodeElement::Accept(_) => char::MAX as usize + 1,
        }
    }
}

impl<M> From<char> for TerminalNodeElement<M> {
    fn from(value: char) -> Self {
        Self::Char(value)
    }
}

impl<M> TerminalNodeElement<M> {
    fn is_nullable(&self) -> bool {
        use TerminalNodeElement::*;
        match self {
            Char(_) | Accept(_) => false,
            Epsilon => true,
        }
    }
}

#[derive(Debug, PartialEq)]
pub(crate) enum TrieNode<M> {
    CatNode(Box<TrieNode<M>>, Box<TrieNode<M>>, TrieMeta),
    StarNode(Box<TrieNode<M>>, TrieMeta),
    OrNode(Box<TrieNode<M>>, Box<TrieNode<M>>, TrieMeta),
    TerminalNode(TerminalNodeElement<M>, TrieMeta, usize),
}

impl<M: std::fmt::Debug> TrieNode<M> {
    pub(crate) fn get_meta(&self) -> &TrieMeta {
        use TrieNode::*;

        match self {
            CatNode(_, _, trie_meta) => trie_meta,
            StarNode(_, trie_meta) => trie_meta,
            OrNode(_, _, trie_meta) => trie_meta,
            TerminalNode(_, trie_meta, _) => trie_meta,
        }
    }
    fn cat(self, new_node: Self) -> Self {
        let meta = TrieMeta::calculate_first_pass_for_cat(self.get_meta(), new_node.get_meta());
        Self::CatNode(Box::new(self), Box::new(new_node), meta)
    }
    fn or(self, new_node: Self) -> Self {
        let meta = TrieMeta::calculate_first_pass_for_or(self.get_meta(), new_node.get_meta());
        Self::OrNode(Box::new(self), Box::new(new_node), meta)
    }
    fn star(self) -> Self {
        let meta = TrieMeta::calculate_first_pass_for_star(self.get_meta());
        Self::StarNode(Box::new(self), meta)
    }

    fn terminal(c: impl Into<TerminalNodeElement<M>>, index: usize) -> Self {
        let c = c.into();
        let meta = TrieMeta::calculate_first_pass_from_char(&c, index);
        Self::TerminalNode(c, meta, index)
    }

    pub(crate) fn build_from_regex(
        regex: &str,
        accept: ConversionFn<M>,
        index: &mut usize,
    ) -> Result<Self> {
        let t =
            TrieNode::from_iterator(&mut regex.chars().map(|x| x.into()).peekable(), index)?.cat(
                TrieNode::<M>::terminal(TerminalNodeElement::Accept(accept.clone()), *index),
            );
        *index += 1;
        Ok(t)
    }

    pub(crate) fn or_from_regex(
        prev: Self,
        regex: &str,
        accept: ConversionFn<M>,
        index: &mut usize,
    ) -> Result<Self> {
        Ok(prev.or(Self::build_from_regex(regex, accept, index)?))
    }

    fn from_iterator<I: Iterator<Item = TerminalNodeElement<M>>>(
        iter: &mut Peekable<I>,
        index: &mut usize,
    ) -> Result<Self> {
        let mut is_escape = false;
        let mut root_node: Option<Self> = None;

        use TerminalNodeElement::*;
        eprintln!("start new");

        while let (Some(next_char), peek) = (iter.next(), iter.peek()) {
            match (&is_escape, next_char, peek) {
                // Escape Section
                (false, Char('\\'), _) => is_escape = true,
                (true, Char('*'), _) => {
                    root_node = Some(
                        root_node
                            .map(|t| t.cat(Self::terminal('*', *index)))
                            .unwrap_or(Self::terminal('*', *index)),
                    );

                    *index += 1;
                    is_escape = false;
                }
                (true, Char('|'), _) => {
                    root_node = Some(
                        root_node
                            .map(|t| t.cat(Self::terminal('|', *index)))
                            .unwrap_or(Self::terminal('|', *index)),
                    );

                    *index += 1;
                    is_escape = false;
                }

                (false, Char('('), _) => {
                    let mut next_tree = Self::from_iterator(iter, index)?;
                    if matches!(iter.peek(), Some(Char('*'))) {
                        next_tree = next_tree.star();
                    }
                    // I can't use the .map(|| ..).unwrap_or(..) pattern cause of
                    // the borrow checker
                    root_node = Some(if let Some(r) = root_node {
                        r.cat(next_tree)
                    } else {
                        next_tree
                    });
                }
                // (false, Char(')'), Some(Char('*'))) => {
                //     root_node = root_node.map(|x| x.star());
                //     break;
                // }
                (false, Char(')'), _) => break,

                (false, Char('*'), _) => {
                    // root_node = Some(
                    //     root_node
                    //         .map(|x| x.star())
                    //         .ok_or(anyhow!("'*' can't be the first character"))?,
                    // )
                }

                (false, Char('|'), _) => {
                    let next_tree = Self::from_iterator(iter, index)?;
                    root_node = Some(
                        root_node
                            .map(|t| t.or(next_tree))
                            .ok_or(anyhow!("'|' can not be the first character"))?,
                    );
                    break;
                }

                (false, x, Some(Char('*'))) => {
                    let new_node = || Self::terminal(x, *index).star();
                    root_node = Some(root_node.map(|t| t.cat(new_node())).unwrap_or(new_node()));
                    *index += 1;
                }

                (false, x, _) => {
                    root_node = Some(
                        root_node
                            .map(|t| t.cat(Self::terminal(x, *index)))
                            .unwrap_or(Self::terminal(x, *index)),
                    );

                    *index += 1;
                }

                (true, _, _) => bail!("Invalid pattern"),
            };
            eprintln!("node: {:?}", root_node);
        }

        let result = root_node.ok_or(anyhow!("Failed to find value"));
        eprintln!("ending: {:?}", result);
        result
    }

    // TODO: Move this into the `from_iterator` function
    // I think I should be able to do this as I go
    // instead of doing another O(n) loop
    pub(crate) fn calculate_follow_pos(&self, size: usize) -> Vec<HashSet<usize>> {
        let mut stack = vec![self];
        let mut follow_pos = vec![HashSet::new(); size];

        while let Some(current_ref) = stack.pop() {
            use TrieNode::*;
            match current_ref {
                CatNode(left, right, _) => {
                    for idx in left.get_meta().last_pos.iter() {
                        follow_pos[*idx].extend(right.get_meta().first_pos.iter().cloned());
                    }

                    stack.push(right);
                    stack.push(left);
                }
                StarNode(node, _) => {
                    for idx in node.get_meta().last_pos.iter() {
                        follow_pos[*idx].extend(node.get_meta().first_pos.iter().cloned());
                    }

                    stack.push(node)
                }
                OrNode(left, right, _) => {
                    stack.push(right);
                    stack.push(left);
                }
                TerminalNode(_, _, _) => {}
            }
        }

        follow_pos
    }

    pub(crate) fn get_refs(&self) -> Vec<(&TrieNode<M>, TerminalNodeElement<M>)> {
        let mut refs = VecDeque::new();
        let mut stack = vec![self];

        while let Some(current_ref) = stack.pop() {
            use TrieNode::*;
            match current_ref {
                CatNode(trie_node, trie_node1, _) | OrNode(trie_node, trie_node1, _) => {
                    stack.push(trie_node.as_ref());
                    stack.push(trie_node1.as_ref());
                }
                StarNode(trie_node, _) => {
                    stack.push(trie_node.as_ref());
                }
                TerminalNode(c, _, _) => {
                    refs.push_front((current_ref, *c));
                }
            }
        }

        refs.into_iter().collect()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn to_string(x: &str) -> Result<String> {
        Ok(x.to_string())
    }

    #[test]
    fn test_str_type() {
        let attempt = Trie::from_regex("c(a|b)*c", to_string).unwrap();

        let correct = TrieNode::terminal('c', 0)
            .cat(
                TrieNode::terminal('a', 1)
                    .or(TrieNode::terminal('b', 2))
                    .star(),
            )
            .cat(TrieNode::terminal('c', 3))
            .cat(TrieNode::terminal(
                TerminalNodeElement::Accept(to_string),
                4,
            ));

        assert_eq!(correct, attempt.root);

        assert_eq!(
            vec![
                HashSet::from([1, 2, 3]),
                HashSet::from([1, 2, 3]),
                HashSet::from([1, 2, 3]),
                HashSet::from([4]),
                HashSet::from([]),
            ],
            attempt.follow_pos
        );
    }

    #[test]
    fn test_paren_a_or_b_star_paren_aab() {
        let attempt = Trie::from_regex("(a|b)*aab", to_string).unwrap();

        let correct = TrieNode::terminal('a', 0)
            .or(TrieNode::terminal('b', 1))
            .star()
            .cat(TrieNode::terminal('a', 2))
            .cat(TrieNode::terminal('a', 3))
            .cat(TrieNode::terminal('b', 4))
            .cat(TrieNode::terminal(
                TerminalNodeElement::Accept(to_string),
                5,
            ));

        assert_eq!(correct, attempt.root);

        assert_eq!(
            attempt.follow_pos,
            vec![
                HashSet::from([0, 1, 2]),
                HashSet::from([0, 1, 2]),
                HashSet::from([3]),
                HashSet::from([4]),
                HashSet::from([5]),
                HashSet::from([])
            ]
        );
    }

    #[test]
    fn test_a_or_b_star_aab() {
        // let attempt: Trie<String> = "a|b*aab".parse().unwrap();
        let attempt = Trie::from_regex("a|b*aab", to_string).unwrap();

        let correct = TrieNode::terminal('a', 0)
            .or(TrieNode::terminal('b', 1)
                .star()
                .cat(TrieNode::terminal('a', 2))
                .cat(TrieNode::terminal('a', 3))
                .cat(TrieNode::terminal('b', 4)))
            .cat(TrieNode::terminal(
                TerminalNodeElement::Accept(to_string),
                5,
            ));

        assert_eq!(correct, attempt.root);

        assert_eq!(
            attempt.follow_pos,
            vec![
                HashSet::from([5]),
                HashSet::from([1, 2]),
                HashSet::from([3]),
                HashSet::from([4]),
                HashSet::from([5]),
                HashSet::from([])
            ]
        );
    }

    #[test]
    fn test_a_or_b_star() {
        // let attempt: Trie<String> = "a|b*".parse().unwrap();
        let attempt = Trie::from_regex("a|b*", to_string).unwrap();

        let correct = TrieNode::terminal('a', 0)
            .or(TrieNode::terminal('b', 1).star())
            .cat(TrieNode::terminal(
                TerminalNodeElement::Accept(to_string),
                2,
            ));

        assert_eq!(correct, attempt.root);

        assert_eq!(
            attempt.follow_pos,
            vec![HashSet::from([2]), HashSet::from([1, 2]), HashSet::from([]),]
        );
    }
}
