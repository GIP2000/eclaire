use anyhow::{anyhow, bail, Result};
use std::{
    collections::{btree_map::Keys, HashSet, VecDeque},
    str::FromStr,
    usize,
};

#[derive(Debug, Default, PartialEq, Clone)]
struct TrieMeta {
    nullable: bool,
    first_pos: HashSet<usize>,
    last_pos: HashSet<usize>,
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

    fn calculate_first_pass_from_char(c: char, index: usize) -> Self {
        let set = HashSet::from([index]);
        Self {
            nullable: c == '\0',
            first_pos: set.clone(),
            last_pos: set,
        }
    }
}

#[derive(Debug)]
struct Trie {
    root: TrieNode,
    follow_pos: Vec<HashSet<usize>>,
    size: usize,
}

#[derive(Debug, PartialEq)]
enum TrieNode {
    CatNode(Box<TrieNode>, Box<TrieNode>, TrieMeta),
    StarNode(Box<TrieNode>, TrieMeta),
    OrNode(Box<TrieNode>, Box<TrieNode>, TrieMeta),
    TerminalNode(char, TrieMeta, usize),
}

impl TrieNode {
    fn get_meta(&self) -> &TrieMeta {
        match self {
            TrieNode::CatNode(_, _, trie_meta) => trie_meta,
            TrieNode::StarNode(_, trie_meta) => trie_meta,
            TrieNode::OrNode(_, _, trie_meta) => trie_meta,
            TrieNode::TerminalNode(_, trie_meta, _) => trie_meta,
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

    fn terminal(c: char, index: usize) -> Self {
        let meta = TrieMeta::calculate_first_pass_from_char(c, index);
        Self::TerminalNode(c, meta, index)
    }

    fn from_iterator(iter: &mut impl Iterator<Item = char>, index: &mut usize) -> Result<Self> {
        let mut is_escape = false;
        let mut root_node: Option<Self> = None;

        while let Some(next_char) = iter.next() {
            match (&is_escape, next_char) {
                // Escape Section
                (false, '\\') => is_escape = true,
                (true, '*') => {
                    root_node = Some(
                        root_node
                            .map(|t| t.cat(Self::terminal('*', *index)))
                            .unwrap_or(Self::terminal('*', *index)),
                    );

                    *index += 1;
                    is_escape = false;
                }
                (true, '|') => {
                    root_node = Some(
                        root_node
                            .map(|t| t.cat(Self::terminal('|', *index)))
                            .unwrap_or(Self::terminal('|', *index)),
                    );

                    *index += 1;
                    is_escape = false;
                }

                (false, '(') => {
                    let next_tree = Self::from_iterator(iter, index)?;

                    // I can't use the .map(|| ..).unwrap_or(..) pattern cause of
                    // the borrow checker
                    root_node = Some(if let Some(r) = root_node {
                        r.cat(next_tree)
                    } else {
                        next_tree
                    });
                }
                (false, ')') => break,

                (false, '*') => {
                    root_node = Some(
                        root_node
                            .map(|x| x.star())
                            .ok_or(anyhow!("'*' can't be the first character"))?,
                    )
                }

                (false, '|') => {
                    let next_tree = Self::from_iterator(iter, index)?;
                    root_node = Some(
                        root_node
                            .map(|t| t.or(next_tree))
                            .ok_or(anyhow!("'|' can not be the first character"))?,
                    );
                    break;
                }

                (false, x) => {
                    root_node = Some(
                        root_node
                            .map(|t| t.cat(Self::terminal(x, *index)))
                            .unwrap_or(Self::terminal(x, *index)),
                    );

                    *index += 1;
                }

                (true, _) => bail!("Invalid pattern"),
            };
        }

        root_node.ok_or(anyhow!("Failed to find value"))
    }

    fn get_refs<'a>(&'a self) -> Vec<&'a TrieNode> {
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
                TerminalNode(_, _, _) => {
                    refs.push_front(current_ref);
                }
            }
        }

        refs.into_iter().collect()
    }

    fn calculate_follow_pos(&self, size: usize) -> Vec<HashSet<usize>> {
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
}

impl FromStr for TrieNode {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_iterator(&mut s.chars(), &mut 0)
    }
}

impl FromStr for Trie {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut size = 0;
        let root = TrieNode::from_iterator(&mut s.chars(), &mut size)?;
        let follow_pos = root.calculate_follow_pos(size);

        Ok(Self {
            root,
            follow_pos,
            size,
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_follow_pos() {
        let attempt: Trie = "(a|b)*aab".parse().unwrap();

        assert_eq!(
            attempt.follow_pos,
            vec![
                HashSet::from([0, 1, 2]),
                HashSet::from([0, 1, 2]),
                HashSet::from([3]),
                HashSet::from([4]),
                HashSet::from([])
            ]
        );
    }

    #[test]
    fn test_paren_a_or_b_star_paren_aab() {
        let attempt: TrieNode = "(a|b)*aab".parse().unwrap();

        let correct = TrieNode::terminal('a', 0)
            .or(TrieNode::terminal('b', 1))
            .star()
            .cat(TrieNode::terminal('a', 2))
            .cat(TrieNode::terminal('a', 3))
            .cat(TrieNode::terminal('b', 4));

        assert_eq!(correct, attempt);
    }

    #[test]
    fn test_a_or_b_star_aab() {
        let attempt: TrieNode = "a|b*aab".parse().unwrap();

        let correct = TrieNode::terminal('a', 0).or(TrieNode::terminal('b', 1)
            .star()
            .cat(TrieNode::terminal('a', 2))
            .cat(TrieNode::terminal('a', 3))
            .cat(TrieNode::terminal('b', 4)));

        assert_eq!(correct, attempt);
    }

    #[test]
    fn test_a_or_b_star() {
        let attempt: TrieNode = "a|b*".parse().unwrap();

        let correct = TrieNode::terminal('a', 0).or(TrieNode::terminal('b', 1).star());

        assert_eq!(correct, attempt);
    }
}
