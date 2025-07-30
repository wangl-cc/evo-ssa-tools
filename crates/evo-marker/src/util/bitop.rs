//! Bitwise operations module.

pub mod u16 {
    const BITS: u32 = u16::BITS;
    const MAX: u16 = u16::MAX;

    /// Generates a bit mask where all bits in the region are 1 and the rest are 0.
    ///
    /// The `low` must be less than `BITS` and `high` must be less than or equal to `BITS`.
    /// Additionally, `low` must be less than `high`.
    #[inline]
    #[must_use]
    pub const fn mask(low: u32, high: u32) -> u16 {
        // debug_assert to avoid overhead on release builds
        debug_assert!(low < BITS);
        debug_assert!(high <= BITS);
        debug_assert!(low < high);
        MAX >> (BITS - (high - low)) << low
    }

    /// Generates a region of bits of given value.
    #[inline]
    #[must_use]
    pub const fn region(low: u32, high: u32, value: u16) -> u16 {
        mask(low, high) & value
    }

    /// Generates a region of bits of given value and move it to the lower bits.
    #[inline]
    #[must_use]
    pub const fn region_and_move(low: u32, high: u32, value: u16) -> u16 {
        region(low, high, value) >> low
    }

    /// Counts how many bits of given value are set in the given region.
    #[inline]
    #[must_use]
    pub const fn ones_in_region(low: u32, high: u32, value: u16) -> u32 {
        region(low, high, value).count_ones()
    }

    /// Applies given mask in the given region to the given value.
    ///
    /// 0 bit is treated as incrementing the value by 1.
    /// 1 bit is treated as decrementing the value by 1.
    #[inline]
    #[must_use]
    pub const fn apply_mask(mask: u16, low: u32, high: u32, value: u32) -> u32 {
        let n_decrement = ones_in_region(low, high, mask);
        let len = high - low;
        value + len - 2 * n_decrement
    }

    /// Reverses the application of given mask in the given region to the given value.
    ///
    /// 0 bit is treated as decrementing the value by 1.
    /// 1 bit is treated as incrementing the value by 1.
    #[inline]
    #[must_use]
    pub const fn reverse_mask(mask: u16, low: u32, high: u32, value: u32) -> u32 {
        let n_increment = ones_in_region(low, high, mask);
        let len = high - low;
        value + 2 * n_increment - len
    }

    #[cfg(test)]
    #[cfg_attr(coverage_nightly, coverage(off))]
    mod tests {
        use super::*;

        #[test]
        fn test_mask() {
            assert_eq!(mask(0, 1), 0b0001);
            assert_eq!(mask(0, 4), 0b1111);
            assert_eq!(mask(1, 2), 0b0010);
        }

        #[test]
        fn test_region() {
            assert_eq!(region(0, 1, 0b1010), 0b0000);
            assert_eq!(region(0, 3, 0b1010), 0b0010);
            assert_eq!(region(0, 4, 0b1010), 0b1010);
            assert_eq!(region(1, 2, 0b1010), 0b0010);
        }

        #[test]
        fn test_region_and_move() {
            assert_eq!(region_and_move(0, 1, 0b1010), 0b0000);
            assert_eq!(region_and_move(0, 3, 0b1010), 0b0010);
            assert_eq!(region_and_move(0, 4, 0b1010), 0b1010);
            assert_eq!(region_and_move(1, 2, 0b1010), 0b0001);
        }

        #[test]
        fn test_ones_in_region() {
            assert_eq!(ones_in_region(0, 1, 0b1010), 0);
            assert_eq!(ones_in_region(0, 3, 0b1010), 1);
            assert_eq!(ones_in_region(0, 4, 0b1010), 2);
            assert_eq!(ones_in_region(1, 2, 0b1010), 1);
        }

        #[test]
        fn test_apply_mask() {
            assert_eq!(apply_mask(0b1010, 0, 3, 1), 2);
            assert_eq!(apply_mask(0b1010, 0, 4, 1), 1);
            assert_eq!(apply_mask(0b1010, 1, 2, 1), 0);
            assert_eq!(apply_mask(0b1010, 1, 3, 1), 1);
        }

        #[test]
        fn test_reverse_mask() {
            assert_eq!(reverse_mask(0b1010, 0, 3, 1), 0);
            assert_eq!(reverse_mask(0b1010, 0, 4, 1), 1);
            assert_eq!(reverse_mask(0b1010, 1, 2, 1), 2);
            assert_eq!(reverse_mask(0b1010, 1, 3, 1), 1);
        }
    }
}

pub mod u4 {
    /// Get a u4 (nibble) from a byte array `data` with given index.
    ///
    /// # Panics
    ///
    /// Panics if `index` is greater than or equal to the `2 * data.len()`.
    pub const fn get(data: &[u8], index: usize) -> u8 {
        let i = index >> 1;
        let packed = data[i];
        let shift = (index & 1) << 2;
        (packed >> shift) & 0xF
    }

    #[cfg(test)]
    #[cfg_attr(coverage_nightly, coverage(off))]
    mod tests {
        use super::*;

        #[test]
        fn test_u4_get() {
            let data = [0x21, 0x43];
            assert_eq!(get(&data, 0), 0x1);
            assert_eq!(get(&data, 1), 0x2);
            assert_eq!(get(&data, 2), 0x3);
            assert_eq!(get(&data, 3), 0x4);
        }
    }
}
