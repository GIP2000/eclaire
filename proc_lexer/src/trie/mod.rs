use anyhow::{anyhow, bail, Result};
use std::str::FromStr;

#[derive(Debug, PartialEq)]
enum Trie {
    CatNode(Box<Trie>, Box<Trie>),
    StarNode(Box<Trie>),
    OrNode(Box<Trie>, Box<Trie>),
    TerminalNode(char),
}

impl Trie {
    fn cat(self, new_node: Self) -> Self {
        Self::CatNode(Box::new(self), Box::new(new_node))
    }
    fn or(self, new_node: Self) -> Self {
        Self::OrNode(Box::new(self), Box::new(new_node))
    }
    fn star(self) -> Self {
        Self::StarNode(Box::new(self))
    }

    fn from_iterator(iter: &mut impl Iterator<Item = char>) -> Result<Self> {
        use Trie::*;
        let mut is_escape = false;
        let mut root_node: Option<Self> = None;

        while let Some(next_char) = iter.next() {
            match (&is_escape, next_char) {
                (false, '\\') => is_escape = true,
                (true, '*') => {
                    root_node = Some(
                        root_node
                            .map(|t| t.cat(TerminalNode('*')))
                            .unwrap_or(TerminalNode('*')),
                    );
                    is_escape = false;
                }
                (true, '|') => {
                    root_node = Some(
                        root_node
                            .map(|t| t.cat(TerminalNode('|')))
                            .unwrap_or(TerminalNode('|')),
                    );
                    is_escape = false;
                }
                (false, '(') => {
                    let next_tree = Self::from_iterator(iter)?;

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
                    let next_tree = Self::from_iterator(iter)?;
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
                            .map(|t| t.cat(TerminalNode(x)))
                            .unwrap_or(TerminalNode(x)),
                    )
                }
                (true, _) => bail!("Invalid pattern"),
            };
        }

        root_node.ok_or(anyhow!("Failed to find value"))
    }
}

impl FromStr for Trie {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_iterator(&mut s.chars())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_paren_a_or_b_star_paren_aab() {
        let attempt: Trie = "(a|b)*aab".parse().unwrap();

        use Trie::*;
        let correct = TerminalNode('a')
            .or(TerminalNode('b'))
            .star()
            .cat(TerminalNode('a'))
            .cat(TerminalNode('a'))
            .cat(TerminalNode('b'));

        assert_eq!(correct, attempt);
    }

    #[test]
    fn test_a_or_b_star_aab() {
        let attempt: Trie = "a|b*aab".parse().unwrap();

        use Trie::*;
        let correct = TerminalNode('a').or(TerminalNode('b')
            .star()
            .cat(TerminalNode('a'))
            .cat(TerminalNode('a'))
            .cat(TerminalNode('b')));

        assert_eq!(correct, attempt);
    }

    #[test]
    fn test_a_or_b_start() {
        let attempt: Trie = "a|b*".parse().unwrap();

        use Trie::*;
        let correct = TerminalNode('a').or(TerminalNode('b').star());

        assert_eq!(correct, attempt);
    }
}
