use std::num::Wrapping;
use std::usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bitset(Wrapping<u32>);

/// Index within 0 ~ 31
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Index32(usize);

impl Index32 {
    pub fn new(index: usize) -> Self {
        debug_assert!(index < 32, "Bitset can hold 0 ~ 31, but received {}", index);
        Index32(index)
    }

    pub fn convert(index: usize, depth: usize) -> Self {
        Index32((index >> (depth * 5)) & 0b11111)
    }

    pub fn max_with(depth: usize) -> Self {
        Index32::convert(usize::MAX, depth)
    }

    pub fn num(&self) -> usize {
        self.0
    }
}

const W0: Wrapping<u32> = Wrapping(0);
const W1: Wrapping<u32> = Wrapping(1);

impl Bitset {
    pub fn new() -> Self {
        Bitset(W0)
    }

    pub fn num(&self) -> u32 {
        (self.0).0
    }

    pub fn get(&self, index: Index32) -> bool {
        self.0 & W1 << index.num() != W0
    }

    pub fn set(&mut self, index: Index32) -> bool {
        let prev = self.get(index);
        self.0 |= W1 << index.num();

        prev
    }

    // NOTE: maybe used later?
    // pub fn unset(&mut self, index: Index32) -> bool {
    //     let prev = self.get(index);
    //     self.0 &= !(W1 << index.num());
    //
    //     prev
    // }

    pub fn packed_index(&self, index: Index32) -> Option<usize> {
        if !self.get(index) {
            return None;
        }
        let index = index.num();

        let leadings_mask = !(W1 << (index + 1) - 1);
        let leadings_count = (self.0 & leadings_mask).0.count_ones();
        Some(leadings_count as usize)
    }

    pub fn iter(&self) -> BitsetIter {
        BitsetIter(*self)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BitsetIter(Bitset);

impl Iterator for BitsetIter {
    type Item = Index32;

    fn next(&mut self) -> Option<Index32> {
        let count = self.0.num().leading_zeros();

        if count == 32 {
            None
        } else {
            Some(Index32(count as usize))
        }
    }
}
