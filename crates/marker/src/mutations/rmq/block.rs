//! Implementation of ±1 RMQ Block

use std::num::Wrapping;

pub(super) type MinAndPos = super::MinAndPos<u8, u8>;

/// The maximum length of ±1 RMQ Block
///
/// Any length larger than 16 is not recommended, as it may lead to excessive cache misses.
pub const MAX_BLOCK_SIZE: u8 = 16;

/// Possible combinations of ±1 RMQ Block
///
/// The number of possible combinations is 2^(MAX_BLOCK_LENGTH - 1), which is 32768.
/// However, if you decrease the block length to 12 or 14, not all combinations will be used.
/// So we can achieve better cache utilization.
pub const BLOCK_COMBINATIONS: usize = 1 << (MAX_BLOCK_SIZE - 1);

/// The length of the ±1 RMQ Block cache
///
/// As each possible minimum value position can be stored in a u4 (two u4 can be packed into a u8).
/// So the cache size is half of the possible combinations.
pub const CACHE_SIZE: usize = BLOCK_COMBINATIONS >> 1;

/// ±1 RMQ Block position cache
///
/// Given a signature of a block, this cache stores the minimum value position of the block.
static BLOCK_RMQ_POS_CACHE: [u8; CACHE_SIZE] = {
    let mut cache = [0; CACHE_SIZE];
    let mut i = 0;
    while i < CACHE_SIZE {
        cache[i] = calc_block_rmq(2 * i as u16);
        cache[i] |= calc_block_rmq(2 * i as u16 + 1) << 4;
        i += 1;
    }
    cache
};

/// Calculate the minimum value of a block with bruteforce
pub const fn calc_block_rmq(sig: u16) -> u8 {
    let mut cur = 0;
    let mut min = cur;
    let mut min_i = 0;
    let mut cur_i = 0;
    while cur_i < MAX_BLOCK_SIZE {
        // Use 0 for increase, 1 for decrease
        // so for block less than 16, we can simply padding the high bits
        // with 0 which will return the same value as the original block
        cur += 1 - 2 * ((sig >> cur_i) & 1) as isize;
        cur_i += 1;
        if cur < min {
            min = cur;
            min_i = cur_i;
        }
    }
    min_i
}

/// Get the position of the minimum value in a block
///
/// This function also works for blocks with length less than 16 as all padding zeros don't
/// affect the result as zero is treated as increasing value and minimum value is always before
/// them.
pub const fn block_rmq(sig: u16) -> u8 {
    nibble_read(&BLOCK_RMQ_POS_CACHE, sig as usize)
}

/// Read a niddle from the data
const fn nibble_read(data: &[u8], index: usize) -> u8 {
    let i = index >> 1;
    let packed = data[i];
    // Always do shift to make avoid branch to make it faster
    (packed >> ((index & 1) << 2)) & 0xF
}

/// RMQ block with length N
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "bitcode", derive(bitcode::Decode, bitcode::Encode))]
pub struct Block {
    sig: u16,
    min_v: u8,
    min_p: u8,
}

impl Block {
    /// Create a block with given length and value at the end of the block.
    fn new(sig: u16, len: u8, v_block_end: u8) -> Self {
        let min_p = block_rmq(sig);
        let min_v = reverse_diff(sig, min_p, len - 1, v_block_end);

        Self { sig, min_v, min_p }
    }

    /// Returns the minimum value of the block and its position.
    pub(crate) fn min_and_pos(self) -> MinAndPos {
        MinAndPos::new(self.min_v, self.min_p)
    }

    /// Get the minimum value of the block from start of the block to the given position.
    ///
    /// Returns the minimum value and its position.
    pub fn min_from_start(self, end: u8) -> MinAndPos {
        debug_assert!(
            end <= MAX_BLOCK_SIZE,
            "End position must be less than or equal to {MAX_BLOCK_SIZE}"
        );

        // If min value in the range of the block, return it
        if end >= self.min_p {
            return MinAndPos::new(self.min_v, self.min_p);
        }

        // Only from given start position to the end should be considered for RMQ
        // so all lower bits are should be masked out
        let sig_from_start = self.sig & ((1 << end) - 1);
        let min_pos = block_rmq(sig_from_start);

        // Calculate the local minimum value from block minimum
        let min_value = reverse_diff(self.sig, min_pos, self.min_p, self.min_v);

        MinAndPos::new(min_value, min_pos)
    }

    /// Get the minimum value of the block from given start position to the end.
    ///
    /// Returns the minimum value and its position.
    pub fn min_to_end(self, start: u8) -> MinAndPos {
        debug_assert!(
            start <= MAX_BLOCK_SIZE,
            "Start position must be less than or equal to {MAX_BLOCK_SIZE}"
        );

        // If min value in the range of the block, return it
        if start <= self.min_p {
            return MinAndPos::new(self.min_v, self.min_p);
        }

        // Only from given start position to the end should be considered for RMQ
        // so all lower bits are should be shifted out
        let sig_to_end = self.sig >> start;
        let min_pos = block_rmq(sig_to_end) + start;

        // Calculate the local minimum value from block minimum
        let min_value = apply_diff(self.sig, self.min_p, min_pos, self.min_v);

        MinAndPos::new(min_value, min_pos)
    }

    pub fn min_in(&self, start: u8, end: u8) -> MinAndPos {
        debug_assert!(
            start <= end,
            "Start position must be less than or equal to end position"
        );
        debug_assert!(
            end <= MAX_BLOCK_SIZE,
            "End position must be less than or equal to {MAX_BLOCK_SIZE}"
        );

        // If min value in the range of the block, return it
        if start <= self.min_p && self.min_p <= end {
            return MinAndPos::new(self.min_v, self.min_p);
        }

        // Only from given start position to the end should be considered for RMQ
        // so all lower bits are should be shifted out
        // and all higher bits are should be masked out
        let sig = (self.sig >> start) & ((1 << (end - start)) - 1);
        let min_pos = block_rmq(sig) + start;

        // Calculate the local minimum value from block minimum
        let min_value = if self.min_p < min_pos {
            apply_diff(self.sig, self.min_p, min_pos, self.min_v)
        } else {
            reverse_diff(self.sig, min_pos, self.min_p, self.min_v)
        };

        MinAndPos::new(min_value, min_pos)
    }
}

/// Apply a change of signature to given value
fn apply_diff(sig: u16, start: u8, end: u8, value: u8) -> u8 {
    let diff_block_len = end - start;
    let mask = (1 << diff_block_len) - 1;
    let diff_sig = (sig >> start) & mask;
    let n_decrease = diff_sig.count_ones() as u8;
    value + diff_block_len - 2 * n_decrease
}

/// Reverse apply a change of signature to given value
fn reverse_diff(sig: u16, start: u8, end: u8, value: u8) -> u8 {
    let diff_block_len = end - start;
    let mask = (1 << diff_block_len) - 1;
    let diff_sig = (sig >> start) & mask;
    let n_decrease = diff_sig.count_ones() as u8;
    value + 2 * n_decrease - diff_block_len
}

pub struct BlockSteper<const N: u8> {
    blocks: Vec<Block>,
    current_block: u16,
    current_position: u32,
    current_depth: Wrapping<u8>,
}

impl<const N: u8> BlockSteper<N> {
    pub fn with_capacity(capacity: usize) -> Self {
        debug_assert!(
            0 < N && N <= MAX_BLOCK_SIZE,
            "Block size must be between 1 and {MAX_BLOCK_SIZE}"
        );

        Self {
            blocks: Vec::with_capacity(capacity / N as usize + 1),
            current_block: 0,
            current_position: 0,
            current_depth: Wrapping(0),
        }
    }

    pub fn step(&mut self, down: bool) {
        self.current_position += 1;

        let current_position = self.current_position;
        let inblock_index = (current_position % (N as u32)) as u8;

        // Each block has N elements, but their signatures is only N - 1 bits
        // So we ignore first step in each block
        if inblock_index == 0 {
            self.blocks
                .push(Block::new(self.current_block, N, self.current_depth.0));
            self.current_block = 0;
        } else {
            self.current_block |= (down as u16) << (inblock_index - 1);
        }

        // Use overflow to do +1 or -1 for unsigned integers
        self.current_depth += Wrapping(1) - Wrapping(2 * down as u8);
    }

    pub(crate) fn finish(self) -> Vec<Block> {
        let inblock_index = (self.current_position % (N as u32)) as u8;
        let mut blocks = self.blocks;

        // Push the last block
        // if the inblock_index == 0, the block is empty, but we have a start value
        blocks.push(Block::new(
            self.current_block,
            inblock_index + 1,
            self.current_depth.0,
        ));

        blocks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_rmq() {
        assert_eq!(calc_block_rmq(0b000_000_000_000_000), 0);
        assert_eq!(calc_block_rmq(0b000_000_000_000_001), 1);
        assert_eq!(calc_block_rmq(0b000_000_000_000_011), 2);
        assert_eq!(calc_block_rmq(0b000_000_000_000_111), 3);
        assert_eq!(calc_block_rmq(0b000_000_000_001_111), 4);
        assert_eq!(calc_block_rmq(0b000_000_000_011_111), 5);
        assert_eq!(calc_block_rmq(0b000_000_000_111_111), 6);
        assert_eq!(calc_block_rmq(0b000_000_001_111_111), 7);
        assert_eq!(calc_block_rmq(0b000_000_011_111_111), 8);
        assert_eq!(calc_block_rmq(0b000_000_111_111_111), 9);
        assert_eq!(calc_block_rmq(0b000_001_111_111_111), 10);
        assert_eq!(calc_block_rmq(0b000_011_111_111_111), 11);
        assert_eq!(calc_block_rmq(0b000_111_111_111_111), 12);
        assert_eq!(calc_block_rmq(0b001_111_111_111_111), 13);
        assert_eq!(calc_block_rmq(0b011_111_111_111_111), 14);
        assert_eq!(calc_block_rmq(0b111_111_111_111_111), 15);

        assert_eq!(calc_block_rmq(0b011_000_000_111_100), 6);
        assert_eq!(calc_block_rmq(0b011_111_100_111_100), 14);

        assert_eq!(block_rmq(0b011_000_000_111_100), 6);
        assert_eq!(block_rmq(0b011_111_100_111_100), 14);
    }

    #[test]
    fn test_apply_diff() {
        assert_eq!(apply_diff(0b000_000_000_000_000, 0, 15, 0), 15);
        assert_eq!(apply_diff(0b000_000_000_000_001, 0, 15, 0), 13);
        assert_eq!(apply_diff(0b000_000_000_000_101, 0, 15, 0), 11);
        assert_eq!(apply_diff(0b000_000_000_110_100, 0, 2, 0), 2);
        assert_eq!(apply_diff(0b000_000_000_110_100, 0, 3, 0), 1);
    }

    #[test]
    fn test_reverse_diff() {
        assert_eq!(reverse_diff(0b000_000_000_000_000, 0, 15, 15), 0);
        assert_eq!(reverse_diff(0b000_000_000_000_001, 0, 15, 15), 2);
        assert_eq!(reverse_diff(0b000_000_000_000_101, 0, 15, 15), 4);
        assert_eq!(reverse_diff(0b000_000_000_110_100, 0, 2, 15), 13);
        assert_eq!(reverse_diff(0b000_000_000_110_100, 0, 3, 15), 14);
    }

    #[test]
    fn test_block_min() {
        let blk: Block = Block::new(0b011_110_111_100_111, 16, 1);
        // Indexes:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, a, b, c, d, e, f]
        // Sequance: [8, 7, 6, 5, 6, 7, 6, 5, 4, 3, 4, 3, 2, 1, 0, 1]

        assert_eq!(blk.min_and_pos(), MinAndPos::new(0, 14));

        assert_eq!(blk.min_from_start(15), MinAndPos::new(0, 14));
        assert_eq!(blk.min_from_start(14), MinAndPos::new(0, 14));
        assert_eq!(blk.min_from_start(13), MinAndPos::new(1, 13));
        assert_eq!(blk.min_from_start(12), MinAndPos::new(2, 12));
        assert_eq!(blk.min_from_start(11), MinAndPos::new(3, 9));
        assert_eq!(blk.min_from_start(10), MinAndPos::new(3, 9));
        assert_eq!(blk.min_from_start(9), MinAndPos::new(3, 9));
        assert_eq!(blk.min_from_start(8), MinAndPos::new(4, 8));
        assert_eq!(blk.min_from_start(7), MinAndPos::new(5, 3));
        assert_eq!(blk.min_from_start(6), MinAndPos::new(5, 3));
        assert_eq!(blk.min_from_start(5), MinAndPos::new(5, 3));
        assert_eq!(blk.min_from_start(4), MinAndPos::new(5, 3));
        assert_eq!(blk.min_from_start(3), MinAndPos::new(5, 3));
        assert_eq!(blk.min_from_start(2), MinAndPos::new(6, 2));
        assert_eq!(blk.min_from_start(1), MinAndPos::new(7, 1));
        assert_eq!(blk.min_from_start(0), MinAndPos::new(8, 0));

        assert_eq!(blk.min_to_end(0), MinAndPos::new(0, 14));
        assert_eq!(blk.min_to_end(1), MinAndPos::new(0, 14));
        assert_eq!(blk.min_to_end(2), MinAndPos::new(0, 14));
        assert_eq!(blk.min_to_end(3), MinAndPos::new(0, 14));
        assert_eq!(blk.min_to_end(4), MinAndPos::new(0, 14));
        assert_eq!(blk.min_to_end(5), MinAndPos::new(0, 14));
        assert_eq!(blk.min_to_end(6), MinAndPos::new(0, 14));
        assert_eq!(blk.min_to_end(7), MinAndPos::new(0, 14));
        assert_eq!(blk.min_to_end(8), MinAndPos::new(0, 14));
        assert_eq!(blk.min_to_end(9), MinAndPos::new(0, 14));
        assert_eq!(blk.min_to_end(10), MinAndPos::new(0, 14));
        assert_eq!(blk.min_to_end(11), MinAndPos::new(0, 14));
        assert_eq!(blk.min_to_end(12), MinAndPos::new(0, 14));
        assert_eq!(blk.min_to_end(13), MinAndPos::new(0, 14));
        assert_eq!(blk.min_to_end(14), MinAndPos::new(0, 14));
        assert_eq!(blk.min_to_end(15), MinAndPos::new(1, 15));

        assert_eq!(blk.min_in(0, 15), MinAndPos::new(0, 14));
        assert_eq!(blk.min_in(0, 13), MinAndPos::new(1, 13));
        assert_eq!(blk.min_in(5, 15), MinAndPos::new(0, 14));
        assert_eq!(blk.min_in(9, 13), MinAndPos::new(1, 13));
        assert_eq!(blk.min_in(13, 15), MinAndPos::new(0, 14));
        assert_eq!(blk.min_in(3, 9), MinAndPos::new(3, 9));
        assert_eq!(blk.min_in(6, 10), MinAndPos::new(3, 9));
        assert_eq!(blk.min_in(11, 15), MinAndPos::new(0, 14));
    }

    #[test]
    fn test_blocks_builder() {
        let mut builder = BlockSteper::<5>::with_capacity(12);

        // The first block, Sequance [0, 1, 0, 1, 2], Signature: 0b0010
        builder.step(false); // 0 -> 1
        builder.step(true); // 1 -> 0
        builder.step(false); // 0 -> 1
        builder.step(false); // 1 -> 2

        // Gap step, two values are in previous block and next block respectively
        // the change will not be included in any blocks' signature
        builder.step(false); // 2 -> 3 (a gap step)

        // The second block, Sequance [3, 4, 3, 4, 3], Signature: 0b1010
        builder.step(false); // 3 -> 4
        builder.step(true); // 4 -> 3
        builder.step(false); // 3 -> 4
        builder.step(true); // 4 -> 3

        // Gap step, two values are in previous block and next block respectively
        // the change will not be included in any blocks' signature
        builder.step(true); // 3 -> 2

        // The third block, Sequance [2, 1, 0], Signature: 0b11
        builder.step(true); // 2 -> 1
        builder.step(true); // 1 -> 0

        let blocks = builder.finish();

        // Verify number of blocks
        assert_eq!(blocks.len(), 3);

        // Verify first block
        assert_eq!(blocks[0].sig, 0b0010);
        assert_eq!(blocks[0].min_and_pos(), MinAndPos::new(0, 0));

        // Verify second block
        assert_eq!(blocks[1].sig, 0b1010);
        assert_eq!(blocks[1].min_and_pos(), MinAndPos::new(3, 0));

        // Verify third (incomplete) block
        assert_eq!(blocks[2].sig, 0b11);
        assert_eq!(blocks[2].min_and_pos(), MinAndPos::new(0, 2));

        let mut builder = BlockSteper::<4>::with_capacity(8);

        // First block, Sequence [0, 1, 2, 1]
        builder.step(false); // 0 -> 1
        builder.step(false); // 1 -> 2
        builder.step(true); // 2 -> 1

        builder.step(false); //  1 -> 2

        // Second block, Sequence [2, 1, 2, 1]
        builder.step(true); // 2 -> 1
        builder.step(false); // 1 -> 2
        builder.step(true); // 2 -> 1

        builder.step(true); // 1 -> 0

        let blocks = builder.finish();

        // Verify number of blocks
        assert_eq!(blocks.len(), 3);

        // Verify first block
        assert_eq!(blocks[0].sig, 0b100);
        assert_eq!(blocks[0].min_and_pos(), MinAndPos::new(0, 0));

        // Verify second block
        assert_eq!(blocks[1].sig, 0b101);
        assert_eq!(blocks[1].min_and_pos(), MinAndPos::new(1, 1));

        // Verify third block
        assert_eq!(blocks[2].sig, 0b000);
        assert_eq!(blocks[2].min_and_pos(), MinAndPos::new(0, 0));
    }
}
