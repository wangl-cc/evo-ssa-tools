//! Block ±1 RMQ Module
//!
//! The Lowest Common Ancestor (LCA) problem in a tree can be reduced to a Range Minimum Query
//! (RMQ). Unlike general RMQ, an LCA-derived RMQ has a key property: consecutive elements differ by
//! only ±1.
//!
//! By partitioning the RMQ sequence into blocks, the problem can be split into two subproblems:
//!
//! 1. Finding the minimum element across blocks within the query range.
//! 2. Finding the minimum element within the two boundary blocks containing the query range.
//!
//! The first subproblem can be solved using a traditional RMQ algorithm on block minima.
//! The second can be efficiently handled by precomputing results for all possible block patterns.
//! For a fixed block length `B`, even if the sequence values vary widely, normalization reveals
//! only `2^(B-1)` unique patterns.
//!
//! The challenge lies in selecting an optimal block length `B` to balance performance.
//! Ideally, `B` should allow both in-block precomputed results and block-wise RMQ structures
//! to fit within the CPU cache for efficiency.
//!
//! Cache considerations for modern CPUs:
//!
//! - L0/L1 data cache: 32KB – 192KB per core
//! - L2/L3 cache: 1MB – 4MB per core or more
//!
//! A block length exceeding 16 is generally inefficient, as its cache grows too large.
//! Conversely, a block length below 8 is suboptimal unless the dataset is very small.
//!
//! The optimal block size depends on cache constraints, data size, and real-world benchmarking.

pub mod block;

pub mod sparse_table;

// use std::fmt::Debug;

use block::{Block, BlockSteper};
use sparse_table::RmqSpareTable;

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "bitcode", derive(bitcode::Decode, bitcode::Encode))]
pub struct MinAndPos<T: Copy + Ord, I: Copy> {
    value: T,
    pos: I,
}

impl<T: Copy + Ord, I: Copy> MinAndPos<T, I> {
    pub fn new(value: T, pos: I) -> Self {
        Self { value, pos }
    }

    pub fn val(&self) -> T {
        self.value
    }

    pub fn pos(&self) -> I {
        self.pos
    }
}

impl<T: Copy + Ord, I: Copy> PartialEq for MinAndPos<T, I> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<T: Copy + Ord, I: Copy> Eq for MinAndPos<T, I> {}

impl<T: Copy + Ord, I: Copy> PartialOrd for MinAndPos<T, I> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Copy + Ord, I: Copy> Ord for MinAndPos<T, I> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.value.cmp(&other.value)
    }
}

#[derive(Debug)]
#[cfg_attr(feature = "bitcode", derive(bitcode::Decode, bitcode::Encode))]
pub struct BlockRMQ<const N: u8> {
    blocks: Vec<Block>,
    spare_table: RmqSpareTable<block::MinAndPos>,
}

impl<const N: u8> BlockRMQ<N> {
    pub fn min_in(&self, start: u32, end: u32) -> u32 {
        debug_assert!(
            start < end,
            "Start index {start} must be less than or equal to end index {end}"
        );

        let start_block = start / N as u32;
        let end_block = end / N as u32;
        let start_in_blk = (start % N as u32) as u8;
        let end_in_blk = (end % N as u32) as u8;

        if start_block == end_block {
            self.blocks[start_block as usize]
                .min_in(start_in_blk, end_in_blk)
                .pos() as u32
                + start_block * N as u32
        } else {
            let start_block_min = self.blocks[start_block as usize].min_to_end(start_in_blk);
            let end_block_min = self.blocks[end_block as usize].min_from_start(end_in_blk);
            let (min, offset) = if start_block_min < end_block_min {
                (start_block_min, start_block * N as u32)
            } else {
                (end_block_min, end_block * N as u32)
            };

            // If no intermediate blocks, return the minimum of the two blocks
            let inter_block_start = start_block + 1;
            let inter_block_end = end_block - 1;

            match inter_block_start.cmp(&inter_block_end) {
                std::cmp::Ordering::Less => {
                    let min_inter = self.spare_table.min_in(start_block + 1, end_block - 1);
                    let min_inter_pos = min_inter.pos();
                    let min_inter_val = min_inter.val();
                    if min_inter_val < min {
                        min_inter_pos + min_inter_val.pos() as u32
                    } else {
                        min.pos() as u32 + offset
                    }
                }
                std::cmp::Ordering::Equal => {
                    let min_inter_pos = inter_block_start;
                    let min_inter_val = self.blocks[min_inter_pos as usize].min_and_pos();
                    if min_inter_val < min {
                        min_inter_pos + min_inter_val.pos() as u32
                    } else {
                        min.pos() as u32 + offset
                    }
                }
                std::cmp::Ordering::Greater => min.pos() as u32 + start_block * N as u32,
            }
        }
    }
}

pub struct BlockRMQSteper<const N: u8> {
    blocks: BlockSteper<N>,
}

impl<const N: u8> BlockRMQSteper<N> {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            blocks: BlockSteper::with_capacity(capacity),
        }
    }

    pub fn step(&mut self, down: bool) {
        self.blocks.step(down);
    }

    pub fn finish(self) -> BlockRMQ<N> {
        let blocks = self.blocks.finish();
        let block_mins = blocks
            .iter()
            .map(|block| block.min_and_pos())
            .collect::<Vec<_>>();
        let spare_table = RmqSpareTable::new(&block_mins);
        BlockRMQ {
            blocks,
            spare_table,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_block_rmq() {
        let steps = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1];
        // Depthes:         [0, 1, 0, 1, 0, 1, 2, 1, 2, 1, 2, 1, 0];

        let mut builder = BlockRMQSteper::<2>::with_capacity(steps.len());
        for &down in &steps {
            builder.step(down == 1);
        }
        let rmq = builder.finish();

        assert_eq!(rmq.min_in(0, 10), 0);
        assert_eq!(rmq.min_in(6, 8), 7);
        assert_eq!(rmq.min_in(6, 7), 7);
        assert_eq!(rmq.min_in(4, 7), 4);
        assert_eq!(rmq.min_in(7, 12), 12);
    }
}
