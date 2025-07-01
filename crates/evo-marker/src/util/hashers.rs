use std::hash::Hasher;

/// A no-op hasher that does not hash anything and only returns the integer value.
///
/// The default hasher in Rust is secure but slow, which is not suitable for some use cases.
/// This hasher is designed for those cases needing a fast hasher (e.g. counting integers).
///
/// This hasher only supports integer hash, for other types like `String`, it will panic.
#[derive(Debug, Default)]
pub(crate) struct NoHashHasher {
    state: u64,
}

impl Hasher for NoHashHasher {
    fn finish(&self) -> u64 {
        self.state
    }

    fn write(&mut self, _: &[u8]) {
        unimplemented!("NoHashHasher only support integer hash")
    }

    // Signed integers will be converted to unsigned in default implementation.
    // So no need to implement write_i8, write_i16, write_i32, write_i64, write_i128 here.

    fn write_u8(&mut self, i: u8) {
        self.write_u64(i as u64)
    }

    fn write_u16(&mut self, i: u16) {
        self.write_u64(i as u64)
    }

    fn write_u32(&mut self, i: u32) {
        self.write_u64(i as u64)
    }

    fn write_u64(&mut self, i: u64) {
        self.state = i;
    }

    fn write_usize(&mut self, i: usize) {
        self.write_u64(i as u64);
    }

    fn write_u128(&mut self, i: u128) {
        self.write_u64(i as u64);
    }
}

/// A `HashMap` that uses `NoHashHasher` as its hasher.
pub(crate) type NoHashMap<K, V> =
    std::collections::HashMap<K, V, std::hash::BuildHasherDefault<NoHashHasher>>;

#[cfg(test)]
mod tests {
    use super::*;

    fn hash<T: std::hash::Hash>(t: &T) -> u64 {
        let mut hasher = NoHashHasher::default();
        t.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn test_integer_hash() {
        // Unsigned integers
        assert_eq!(hash(&1u8), 1);
        assert_eq!(hash(&2u16), 2);
        assert_eq!(hash(&3u32), 3);
        assert_eq!(hash(&4u64), 4);
        assert_eq!(hash(&4usize), 4);
        assert_eq!(hash(&5u128), 5);

        // Signed integers
        assert_eq!(hash(&1i8), 1);
        assert_eq!(hash(&2i16), 2);
        assert_eq!(hash(&3i32), 3);
        assert_eq!(hash(&4i64), 4);
        assert_eq!(hash(&4isize), 4);
        assert_eq!(hash(&5i128), 5);
    }

    #[test]
    #[should_panic]
    fn test_no_integer_hash() {
        hash(&[1, 2, 3, 4]);
    }
}
