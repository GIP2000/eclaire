use crate::utils::VecSet;
use crate::{dfa::DFA_SIZE, AcceptFunc};
use std::{collections::VecDeque, fmt::Debug, iter::Peekable};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TrieError {
    #[error("Invalid start character {0}")]
    InvalidStart(u8),
    #[error("Empty Regex")]
    EmptyRegex,
    #[error("Invalid escape parameter {0}")]
    InvalidEscapeParameter(usize),
    #[error("Invalid range pattern it must be [a-b]")]
    InvalidRangePattern,
    #[error("Invalid empty bracket in regex")]
    EmptyBrackets,
}

type Result<T> = std::result::Result<T, TrieError>;

#[derive(Debug, Default, PartialEq, Clone)]
pub(crate) struct TrieMeta {
    pub(crate) nullable: bool,
    pub(crate) first_pos: VecSet<usize>,
    pub(crate) last_pos: VecSet<usize>,
}

impl TrieMeta {
    fn calculate_first_pass_for_cat(l: &Self, r: &Self) -> Self {
        let first_pos = if l.nullable {
            l.first_pos.union(&r.first_pos).cloned().collect()
        } else {
            l.first_pos.clone()
        };

        let last_pos = if r.nullable {
            l.last_pos.union(&r.last_pos).cloned().collect()
        } else {
            r.last_pos.clone()
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

    fn calculate_first_pass_from_char<A: AcceptFunc>(
        c: &TerminalNodeElement<A>,
        index: usize,
    ) -> Self {
        let set = VecSet::from([index]);
        Self {
            nullable: c.is_nullable(),
            first_pos: set.clone(),
            last_pos: set,
        }
    }
}

#[derive(PartialEq)]
pub(crate) struct Trie<A: AcceptFunc> {
    pub(crate) root: TrieNode<A>,
    pub(crate) follow_pos: Vec<VecSet<usize>>,
    pub(crate) size: usize,
}

impl<A: AcceptFunc> std::fmt::Debug for Trie<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Trie")
            .field("root", &self.root)
            .field("follow_pos", &self.follow_pos)
            .field("size", &self.size)
            .finish()
    }
}

impl<A: AcceptFunc + Eq> Trie<A> {
    pub(crate) fn from_regex(regex: &str, accept: A) -> Result<Self> {
        let mut size = 0;

        let a = TerminalNodeElement::Accept(accept.clone(), 0);
        let root: TrieNode<A> =
            TrieNode::from_iterator(&mut regex.bytes().map(|x| x.into()).peekable(), &mut size)?
                // Add the accept state to the end
                .cat(TrieNode::terminal(
                    TerminalNodeElement::Accept(accept, 0),
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

#[derive(Clone, PartialEq, Eq)]
pub(crate) enum TerminalNodeElement<A: AcceptFunc> {
    Char(u8),
    Accept(A, usize),
}

impl<A: AcceptFunc> Debug for TerminalNodeElement<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Char(arg0) => {
                let readable = if arg0.is_ascii_alphanumeric() {
                    &(*arg0 as char) as &dyn std::fmt::Debug
                } else {
                    arg0 as &dyn std::fmt::Debug
                };

                f.debug_tuple("Char").field(readable).finish()
            }
            Self::Accept(_, rank) => f.debug_tuple("Accept").field(rank).finish(),
        }
    }
}

impl<A: AcceptFunc> From<TerminalNodeElement<A>> for usize {
    fn from(value: TerminalNodeElement<A>) -> Self {
        match value {
            TerminalNodeElement::Char(x) => x as usize,
            TerminalNodeElement::Accept(_, _) => char::MAX as usize + 1,
        }
    }
}

impl<A: AcceptFunc> From<u8> for TerminalNodeElement<A> {
    fn from(value: u8) -> Self {
        Self::Char(value)
    }
}

impl<A: AcceptFunc> TerminalNodeElement<A> {
    #[inline(always)]
    fn is_nullable(&self) -> bool {
        false
    }
}

#[derive(PartialEq)]
pub(crate) enum TrieNode<A>
where
    A: AcceptFunc,
{
    CatNode(Box<TrieNode<A>>, Box<TrieNode<A>>, TrieMeta),
    StarNode(Box<TrieNode<A>>, TrieMeta),
    OrNode(Box<TrieNode<A>>, Box<TrieNode<A>>, TrieMeta),
    TerminalNode(TerminalNodeElement<A>, TrieMeta, usize),
}

impl<A: AcceptFunc> Debug for TrieNode<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fn fmt_with_ident<A: AcceptFunc>(
            value: &TrieNode<A>,
            f: &mut std::fmt::Formatter<'_>,
            indent_level: u32,
        ) -> std::fmt::Result {
            macro_rules! indent {
                () => {
                    for _ in 0..indent_level {
                        write!(f, "\t")?;
                    }
                };

                ($add: expr) => {
                    for _ in 0..(indent_level + $add) {
                        write!(f, "\t")?;
                    }
                };
            }

            indent!();

            enum Flat<'a, A: AcceptFunc> {
                Single(&'a TrieNode<A>),
                Double(&'a TrieNode<A>, &'a TrieNode<A>),
                Term(&'a TerminalNodeElement<A>, usize),
            }

            let (inner, meta) = match value {
                TrieNode::CatNode(trie_node, trie_node1, trie_meta) => {
                    writeln!(f, "CatNode(")?;
                    (Flat::Double(trie_node.as_ref(), trie_node1), trie_meta)
                }
                TrieNode::StarNode(trie_node, trie_meta) => {
                    writeln!(f, "StarNode(")?;
                    (Flat::Single(trie_node), trie_meta)
                }
                TrieNode::OrNode(trie_node, trie_node1, trie_meta) => {
                    writeln!(f, "OrNode(")?;
                    (Flat::Double(trie_node, trie_node1), trie_meta)
                }
                TrieNode::TerminalNode(terminal_node_element, trie_meta, idx) => {
                    writeln!(f, "TerminalNode(")?;
                    (Flat::Term(terminal_node_element, *idx), trie_meta)
                }
            };

            match inner {
                Flat::Single(trie_node) => {
                    fmt_with_ident(trie_node, f, indent_level + 1)?;
                }
                Flat::Double(trie_node, trie_node1) => {
                    fmt_with_ident(trie_node, f, indent_level + 1)?;
                    fmt_with_ident(trie_node1, f, indent_level + 1)?;
                }
                Flat::Term(terminal_node_element, idx) => {
                    indent!(1);
                    writeln!(f, "{:?}, {:?}", terminal_node_element, idx)?;
                }
            }

            indent!(1);
            writeln!(f, "{:?}", meta)?;
            indent!();
            writeln!(f, ")")
        }

        fmt_with_ident(self, f, 0)
    }
}

impl<A: AcceptFunc + Eq> TrieNode<A> {
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

    fn terminal(c: impl Into<TerminalNodeElement<A>>, index: usize) -> Self {
        let c = c.into();
        let meta = TrieMeta::calculate_first_pass_from_char(&c, index);
        Self::TerminalNode(c, meta, index)
    }

    pub(crate) fn build_from_regex(
        regex: &str,
        accept: A,
        index: &mut usize,
        rank: usize,
    ) -> Result<Self> {
        let t = Self::from_iterator(&mut regex.bytes().map(|x| x.into()).peekable(), index)?.cat(
            Self::terminal(TerminalNodeElement::Accept(accept.clone(), rank), *index),
        );
        *index += 1;
        Ok(t)
    }

    pub(crate) fn or_from_regex(
        prev: Self,
        regex: &str,
        accept: A,
        index: &mut usize,
        rank: usize,
    ) -> Result<Self> {
        Ok(prev.or(Self::build_from_regex(regex, accept, index, rank)?))
    }

    fn handle_bracket<I: Iterator<Item = TerminalNodeElement<A>>>(
        iter: &mut Peekable<I>,
        index: &mut usize,
    ) -> Result<Self> {
        use TerminalNodeElement::*;

        let is_not = iter.peek().is_some_and(|x| matches!(x, Char(b'^')));

        if is_not {
            iter.next()
                .expect("Unreachable I just peeked and found a ^");
        }

        let mut letters = VecSet::new();

        while let (Some(next_char), peek) = (iter.next(), iter.peek()) {
            match (next_char, peek) {
                (
                    Char(b'\\'),
                    Some(Char(b'^')) | Some(Char(b'-')) | Some(Char(b'[')) | Some(Char(b']')),
                ) => {
                    // normal with escaped chars
                    let value = iter.next().expect("I checked with the peek");
                    letters.insert(value);
                }
                (Char(b']'), _) => {
                    if is_not {
                        return (0..=u8::MAX)
                            .filter_map(|x| {
                                let node = Char(x);
                                (!letters.contains(&node)).then(|| {
                                    let result = Self::terminal(node, *index);
                                    *index += 1;
                                    result
                                })
                            })
                            .reduce(|acc, val| acc.or(val))
                            .ok_or(TrieError::EmptyBrackets);
                    } else {
                        return letters
                            .into_iter()
                            .map(|x| {
                                let result = Self::terminal(x, *index);
                                *index += 1;
                                result
                            })
                            .reduce(|acc, val| acc.or(val))
                            .ok_or(TrieError::EmptyBrackets);
                    }
                } // end
                (a, Some(Char(b'-'))) => {
                    let _ = iter.next().expect("I checked with the peek");
                    let b = iter.next().ok_or(TrieError::InvalidRangePattern)?;

                    let (a, b) = match (a, b) {
                        (Char(a), Char(b)) => (a, b),
                        _ => unreachable!("Error: Invalid token on other side of -"),
                    };

                    letters.extend((a..=b).map(|x| Char(x)));
                } // range
                (value, _) => {
                    letters.insert(value);
                }
            };
        }

        Err(TrieError::InvalidRangePattern)
    }

    fn from_iterator<I: Iterator<Item = TerminalNodeElement<A>>>(
        iter: &mut Peekable<I>,
        index: &mut usize,
    ) -> Result<Self> {
        let mut is_escape = false;
        let mut root_node: Option<Self> = None;

        use TerminalNodeElement::*;

        while let (Some(next_char), peek) = (iter.next(), iter.peek()) {
            match (&is_escape, next_char, peek) {
                // Escape Section
                (false, Char(b'\\'), _) => is_escape = true,
                (true, Char(b'\\'), _) => {
                    root_node = Some(
                        root_node
                            .map(|t| t.cat(Self::terminal(b'\\', *index)))
                            .unwrap_or(Self::terminal(b'\\', *index)),
                    );

                    *index += 1;
                    is_escape = false;
                }
                (true, Char(b'.'), _) => {
                    root_node = Some(
                        root_node
                            .map(|t| t.cat(Self::terminal(b'.', *index)))
                            .unwrap_or(Self::terminal(b'.', *index)),
                    );

                    *index += 1;
                    is_escape = false;
                }
                (true, Char(b'['), _) => {
                    root_node = Some(
                        root_node
                            .map(|t| t.cat(Self::terminal(b'[', *index)))
                            .unwrap_or(Self::terminal(b'[', *index)),
                    );

                    *index += 1;
                    is_escape = false;
                }
                (true, Char(b']'), _) => {
                    root_node = Some(
                        root_node
                            .map(|t| t.cat(Self::terminal(b']', *index)))
                            .unwrap_or(Self::terminal(b']', *index)),
                    );

                    *index += 1;
                    is_escape = false;
                }

                (true, Char(b'*'), _) => {
                    root_node = Some(
                        root_node
                            .map(|t| t.cat(Self::terminal(b'*', *index)))
                            .unwrap_or(Self::terminal(b'*', *index)),
                    );

                    *index += 1;
                    is_escape = false;
                }
                (true, Char(b'|'), _) => {
                    root_node = Some(
                        root_node
                            .map(|t| t.cat(Self::terminal(b'|', *index)))
                            .unwrap_or(Self::terminal(b'|', *index)),
                    );

                    *index += 1;
                    is_escape = false;
                }
                (true, Char(b'('), _) => {
                    root_node = Some(
                        root_node
                            .map(|t| t.cat(Self::terminal(b'(', *index)))
                            .unwrap_or(Self::terminal(b'(', *index)),
                    );

                    *index += 1;
                    is_escape = false;
                }
                (true, Char(b')'), _) => {
                    root_node = Some(
                        root_node
                            .map(|t| t.cat(Self::terminal(b')', *index)))
                            .unwrap_or(Self::terminal(b')', *index)),
                    );

                    *index += 1;
                    is_escape = false;
                }

                (false, Char(b'['), _) => {
                    let mut next_tree = Self::handle_bracket(iter, index)?;
                    if matches!(iter.peek(), Some(Char(b'*'))) {
                        next_tree = next_tree.star();
                    }

                    root_node = Some(match root_node {
                        Some(t) => t.cat(next_tree),
                        None => next_tree,
                    });
                }
                (false, Char(b'('), _) => {
                    let mut next_tree = Self::from_iterator(iter, index)?;
                    if matches!(iter.peek(), Some(Char(b'*'))) {
                        next_tree = next_tree.star();
                    }

                    root_node = Some(match root_node {
                        Some(r) => r.cat(next_tree),
                        None => next_tree,
                    });
                }
                (false, Char(b')'), _) => break,
                (false, Char(b'*'), _) => {}
                (false, Char(b'|'), _) => {
                    let next_tree = Self::from_iterator(iter, index)?;
                    root_node = Some(
                        root_node
                            .map(|t| t.or(next_tree))
                            .ok_or(TrieError::InvalidStart(b'|'))?,
                    );
                    break;
                }

                (false, Char(b'.'), Some(Char(b'*'))) => {
                    let mut node = Self::terminal(0, *index);
                    *index += 1;
                    for a in 1..DFA_SIZE {
                        node = node.or(Self::terminal(TerminalNodeElement::Char(a as u8), *index));
                        *index += 1;
                    }

                    root_node = Some(match root_node {
                        Some(t) => t.cat(node.star()),
                        None => node.star(),
                    });
                }

                (false, x, Some(Char(b'*'))) => {
                    let node = Self::terminal(x.clone(), *index);
                    *index += 1;

                    root_node = Some(match root_node {
                        Some(t) => t.cat(node.star()),
                        None => node.star(),
                    });
                }

                (false, Char(b'.'), _) => {
                    let mut node = Self::terminal(0, *index);
                    *index += 1;
                    for a in 1..DFA_SIZE {
                        node = node.or(Self::terminal(TerminalNodeElement::Char(a as u8), *index));
                        *index += 1;
                    }

                    root_node = Some(match root_node {
                        Some(t) => t.cat(node),
                        None => node,
                    });
                }

                (false, x, _) => {
                    let node = Self::terminal(x.clone(), *index);
                    *index += 1;

                    root_node = Some(match root_node {
                        Some(t) => t.cat(node),
                        None => node,
                    });
                }

                (true, x, _) => return Err(TrieError::InvalidEscapeParameter(x.into())),
            };
        }

        root_node.ok_or(TrieError::EmptyRegex)
    }

    pub(crate) fn calculate_follow_pos(&self, size: usize) -> Vec<VecSet<usize>> {
        let mut stack = vec![self];
        let mut follow_pos = vec![VecSet::new(); size];

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

    pub(crate) fn get_refs(&self) -> Vec<(&TrieNode<A>, TerminalNodeElement<A>)> {
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
                    refs.push_front((current_ref, c.clone()));
                }
            }
        }

        refs.into_iter().collect()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    impl AcceptFunc for &str {
        type Error = Box<dyn std::error::Error>;
        type Output<'a> = &'a str;

        fn convert<'a>(
            &self,
            input: &'a str,
        ) -> std::result::Result<Self::Output<'a>, Self::Error> {
            Ok(input.into())
        }
    }

    #[test]
    fn test_str_type() {
        let attempt = Trie::from_regex("c(a|b)*c", "to_string").unwrap();

        let correct = TrieNode::terminal(b'c', 0)
            .cat(
                TrieNode::terminal(b'a', 1)
                    .or(TrieNode::terminal(b'b', 2))
                    .star(),
            )
            .cat(TrieNode::terminal(b'c', 3))
            .cat(TrieNode::terminal(
                TerminalNodeElement::Accept("to_string", 0),
                4,
            ));

        assert_eq!(correct, attempt.root);

        assert_eq!(
            vec![
                VecSet::from([1, 2, 3]),
                VecSet::from([1, 2, 3]),
                VecSet::from([1, 2, 3]),
                VecSet::from([4]),
                VecSet::from([]),
            ],
            attempt.follow_pos
        );
    }

    #[test]
    fn test_paren_a_or_b_star_paren_aab() {
        let attempt = Trie::from_regex("(a|b)*aab", "to_string").unwrap();

        let correct = TrieNode::terminal(b'a', 0)
            .or(TrieNode::terminal(b'b', 1))
            .star()
            .cat(TrieNode::terminal(b'a', 2))
            .cat(TrieNode::terminal(b'a', 3))
            .cat(TrieNode::terminal(b'b', 4))
            .cat(TrieNode::terminal(
                TerminalNodeElement::Accept("to_string", 0),
                5,
            ));

        assert_eq!(correct, attempt.root);

        assert_eq!(
            attempt.follow_pos,
            vec![
                VecSet::from([0, 1, 2]),
                VecSet::from([0, 1, 2]),
                VecSet::from([3]),
                VecSet::from([4]),
                VecSet::from([5]),
                VecSet::from([])
            ]
        );
    }

    #[test]
    fn test_a_or_b_star_aab() {
        // let attempt: Trie<String> = "a|b*aab".parse().unwrap();
        let attempt = Trie::from_regex("a|b*aab", "to_string").unwrap();

        let correct = TrieNode::terminal(b'a', 0)
            .or(TrieNode::terminal(b'b', 1)
                .star()
                .cat(TrieNode::terminal(b'a', 2))
                .cat(TrieNode::terminal(b'a', 3))
                .cat(TrieNode::terminal(b'b', 4)))
            .cat(TrieNode::terminal(
                TerminalNodeElement::Accept("to_string".into(), 0),
                5,
            ));

        assert_eq!(correct, attempt.root);

        assert_eq!(
            attempt.follow_pos,
            vec![
                VecSet::from([5]),
                VecSet::from([1, 2]),
                VecSet::from([3]),
                VecSet::from([4]),
                VecSet::from([5]),
                VecSet::from([])
            ]
        );
    }

    #[test]
    fn test_a_or_b_star() {
        // let attempt: Trie<String> = "a|b*".parse().unwrap();
        let attempt = Trie::from_regex("a|b*", "to_string").unwrap();

        let correct = TrieNode::terminal(b'a', 0)
            .or(TrieNode::terminal(b'b', 1).star())
            .cat(TrieNode::terminal(
                TerminalNodeElement::Accept("to_string".into(), 0),
                2,
            ));

        assert_eq!(correct, attempt.root);

        assert_eq!(
            attempt.follow_pos,
            vec![VecSet::from([2]), VecSet::from([1, 2]), VecSet::from([]),]
        );
    }

    #[test]

    fn test_log_or() {
        let attempt = Trie::from_regex("\\|\\|", "to_string").unwrap();

        let correct = TrieNode::terminal(b'|', 0)
            .cat(TrieNode::terminal(b'|', 1))
            .cat(TrieNode::terminal(
                TerminalNodeElement::Accept("to_string", 0),
                2,
            ));

        assert_eq!(correct, attempt.root);
    }

    #[test]
    fn test_bracket() {
        let attempt = Trie::from_regex("[a-d]", "to_string").unwrap();

        let correct = TrieNode::terminal(b'a', 0)
            .or(TrieNode::terminal(b'b', 1))
            .or(TrieNode::terminal(b'c', 2))
            .or(TrieNode::terminal(b'd', 3))
            .cat(TrieNode::terminal(
                TerminalNodeElement::Accept("to_string", 0),
                4,
            ));
        assert_eq!(correct, attempt.root);
    }

    #[test]
    fn test_bracket_star() {
        let attempt = Trie::from_regex("[a-d]*", "to_string").unwrap();

        let correct = TrieNode::terminal(b'a', 0)
            .or(TrieNode::terminal(b'b', 1))
            .or(TrieNode::terminal(b'c', 2))
            .or(TrieNode::terminal(b'd', 3))
            .star()
            .cat(TrieNode::terminal(
                TerminalNodeElement::Accept("to_string", 0),
                4,
            ));
        assert_eq!(correct, attempt.root);
    }

    #[test]
    fn test_mult() {
        let attempt = Trie::from_regex("\\*", "to_string").unwrap();

        let correct = TrieNode::terminal(b'*', 0).cat(TrieNode::terminal(
            TerminalNodeElement::Accept("to_string", 0),
            1,
        ));

        assert_eq!(correct, attempt.root);
    }
}
