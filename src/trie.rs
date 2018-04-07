//! Array-Mapped Trie implementation
//!
//! Provides O(log32 n) ~= O(1) index/update

use std::sync::Arc;
use std::mem::size_of;
use std::iter::FromIterator;

use bitset::{Bitset, Index32};

const MAX_DEPTH: usize = (size_of::<usize>() - 1) / 5 + 1;

#[derive(Debug)]
pub struct Trie<T> {
    root: Option<Node<T>>,
}

#[derive(Debug)]
pub struct TrieMut<T> {
    root: NodeMut<T>,
}

#[derive(Debug)]
enum Node<T> {
    One {
        index: usize,
        value: T,
    },
    More {
        bitset: Bitset,
        nodes: Arc<[Node<T>]>,
    },
}

#[derive(Debug)]
enum NodeMut<T> {
    Empty,
    Imut(Node<T>),
    MoreMut(Vec<(Index32, NodeMut<T>)>),
}

use self::{Node::*, NodeMut::*};

impl<T: Clone> Trie<T> {
    pub fn new() -> Self {
        Trie {
            root: None,
        }
    }

    pub fn len(&self) -> usize {
        self.root.as_ref().map_or(0, |node| node.len())
    }

    pub fn get(&self, index: usize) -> Option<T> {
        self.root.as_ref().and_then(|node| node.get(index))
    }

    pub fn to_mut(&self) -> TrieMut<T> {
        TrieMut {
            root: match self.root {
                Some(ref node) => Imut(node.clone()),
                None => Empty,
            },
        }
    }

    pub fn update_all<I: IntoIterator<Item=(usize, T)>>(&self, iter: I) -> Self {
        let mut iter = iter.into_iter();

        match iter.next() {
            None => self.clone(),
            Some((index, value)) => {
                let mut trie = self.to_mut();
                trie.insert(index, value);

                for (index, value) in iter {
                    trie.insert(index, value);
                }

                trie.into_trie()
            }
        }
    }

    pub fn update(&self, index: usize, value: T) -> Self {
        let mut trie = self.to_mut();
        trie.insert(index, value);
        trie.into_trie()
    }

    pub fn remove_all<I: IntoIterator<Item=usize>>(&self, iter: I) -> Self {
        let mut iter = iter.into_iter();

        match iter.next() {
            None => self.clone(),
            Some(index) => {
                let mut trie = self.to_mut();
                trie.remove(index);

                for index in iter {
                    trie.remove(index);
                }

                trie.into_trie()
            }
        }
    }

    pub fn remove(&self, index: usize) -> Self {
        let mut trie = self.to_mut();
        trie.remove(index);
        trie.into_trie()
    }
}

impl<T: Clone> Default for Trie<T> {
    fn default() -> Self {
        Trie::new()
    }
}

impl<T: Clone> Clone for Trie<T> {
    fn clone(&self) -> Self {
        Trie {
            root: self.root.clone(),
        }
    }
}

impl<T: Clone> FromIterator<(usize, T)> for Trie<T> {
    fn from_iter<I: IntoIterator<Item=(usize, T)>>(iter: I) -> Self {
        let trie: TrieMut<T> = iter.into_iter().collect();
        trie.into_trie()
    }
}

impl<T: Clone> TrieMut<T> {
    pub fn new() -> Self {
        TrieMut {
            root: Empty,
        }
    }

    pub fn insert(&mut self, index: usize, value: T) -> Option<T> {
        self.root.insert(0, index, value)
    }

    pub fn remove(&mut self, index: usize) -> Option<T> {
        self.root.remove(0, index)
    }

    pub fn into_trie(self) -> Trie<T> {
        Trie {
            root: self.root.into_node(),
        }
    }
}

impl<T: Clone> Default for TrieMut<T> {
    fn default() -> Self {
        TrieMut::new()
    }
}

impl<T: Clone> Extend<(usize, T)> for TrieMut<T> {
    fn extend<I: IntoIterator<Item=(usize, T)>>(&mut self, iter: I) {
        for (index, value) in iter {
            self.insert(index, value);
        }
    }
}

impl<T: Clone> FromIterator<(usize, T)> for TrieMut<T> {
    fn from_iter<I: IntoIterator<Item=(usize, T)>>(iter: I) -> Self {
        let mut trie = Self::default();
        trie.extend(iter);
        trie
    }
}

impl<T: Clone> Node<T> {
    fn len(&self) -> usize {
        match *self {
            One { .. } => 1,
            More { ref nodes, .. } => nodes.iter().map(|v| v.len()).sum(),
        }
    }

    fn get(&self, index: usize) -> Option<T> {
        let query = index;

        match *self {
            One { ref value, .. } => Some(value.clone()),
            More { bitset, ref nodes } => {
                bitset.packed_index(Index32::new(query % 32))
                    .and_then(|idx| nodes[idx].get(query / 32))
            }
        }
    }
}

impl<T: Clone> Clone for Node<T> {
    fn clone(&self) -> Self {
        match *self {
            One { index, ref value } => One { index, value: value.clone() },
            More { bitset, ref nodes } => More { bitset, nodes: nodes.clone() },
        }
    }
}

fn make_mut<T: Clone>(bitset: Bitset, nodes: &[Node<T>]) -> Vec<(Index32, NodeMut<T>)> {
    bitset.iter()
        .zip(nodes)
        .map(|(idx, value)| (idx, Imut(value.clone())))
        .collect()
}

impl<T: Clone> NodeMut<T> {
    fn insert(&mut self, depth: usize, new_index: usize, new_value: T) -> Option<T> {
        if depth > MAX_DEPTH {
            return None;
        }

        let transform = |idx| Index32::convert(idx, depth);
        let mut res = None;

        let replace = match *self {
            Empty => Some(Imut(One { index: new_index, value: new_value })),
            Imut(One { index, ref mut value }) if index == new_index => {
                res = Some(value.clone());
                Some(Imut(One { index, value: new_value }))
            }
            Imut(One { index, ref mut value }) => {
                let mut pairs = vec![
                    (transform(index), Imut(One {
                        index,
                        value: value.clone(),
                    })),
                    (transform(new_index), Imut(One {
                        index: new_index,
                        value: new_value,
                    })),
                ];

                if index > new_index {
                    pairs.swap(0, 1);
                }

                Some(MoreMut(pairs))
            }
            Imut(More { bitset, ref mut nodes }) => {
                let mut node = MoreMut(make_mut(bitset, &nodes));
                res = node.insert(depth, new_index, new_value);
                Some(node)
            }
            MoreMut(ref mut pairs) => {
                match pairs.binary_search_by_key(&transform(new_index), |p| p.0) {
                    Ok(idx) => {
                        res = pairs[idx].1.insert(depth + 1, new_index, new_value);
                    }
                    Err(idx) => {
                        let index32 = transform(new_index);
                        let node = Imut(One {
                            index: new_index,
                            value: new_value,
                        });
                        pairs.insert(idx, (index32, node));
                    }
                }

                None
            }
        };

        if let Some(replace) = replace {
            *self = replace;
        }

        res
    }

    fn remove(&mut self, depth: usize, del_index: usize) -> Option<T> {
        if depth > MAX_DEPTH {
            return None;
        }

        let transform = |idx| Index32::convert(idx, depth);
        let mut res = None;

        let replace = match *self {
            Empty => None,
            Imut(One { index, ref mut value }) if index == del_index => {
                res = Some(value.clone());
                Some(Empty)
            }
            Imut(One { .. }) => None,
            Imut(More { bitset, ref mut nodes }) => {
                if bitset.get(transform(del_index)) {
                    let mut node = MoreMut(make_mut(bitset, &nodes));
                    res = node.remove(depth, del_index);

                    Some(node)
                } else {
                    None
                }
            }
            MoreMut(ref mut pairs) => {
                match pairs.binary_search_by_key(&transform(del_index), |p| p.0) {
                    Err(_) => {},
                    Ok(idx) => {
                        res = pairs[idx].1.remove(depth + 1, del_index);

                        if let (_, Empty) = pairs[idx] {
                            pairs.remove(idx);
                        }
                    }
                }

                if pairs.len() == 1 {
                    if let (_, Imut(One { index, ref value })) = pairs[0] {
                        Some(Imut(One { index, value: value.clone() }))
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
        };

        if let Some(replace) = replace {
            *self = replace;
        }

        res
    }

    fn into_node(self) -> Option<Node<T>> {
        match self {
            Empty => None,
            Imut(node) => Some(node),
            MoreMut(pairs) => {
                let mut bitset = Bitset::new();
                let mut nodes = Vec::new();

                for (idx32, node_mut) in pairs {
                    bitset.set(idx32);

                    if let Some(node) = node_mut.into_node() {
                        nodes.push(node);
                    }
                }

                Some(More {
                    bitset,
                    nodes: Arc::from(&nodes[..]),
                })
            }
        }
    }
}
