/// A block of canonical keys stored back to back.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Keys {
    bytes: Vec<u8>,
    ends: Vec<usize>,
}

impl Keys {
    pub(crate) fn with_capacity(items: usize, bytes: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(bytes),
            ends: Vec::with_capacity(items),
        }
    }

    pub(crate) fn push_parts(&mut self, head: &[u8], tail: &[u8]) {
        self.bytes.extend_from_slice(head);
        self.bytes.extend_from_slice(tail);
        self.ends.push(self.bytes.len());
    }

    /// Number of keys.
    pub fn len(&self) -> usize {
        self.ends.len()
    }

    /// Whether the block has no keys.
    pub fn is_empty(&self) -> bool {
        self.ends.is_empty()
    }

    /// Key `index`, without the namespace prefix.
    pub fn get(&self, index: usize) -> &[u8] {
        let start = if index == 0 { 0 } else { self.ends[index - 1] };
        &self.bytes[start..self.ends[index]]
    }

    pub(crate) fn select(&self, items: &[usize]) -> Self {
        let mut keys = Self::with_capacity(items.len(), 0);
        for &item in items {
            keys.push_parts(self.get(item), &[]);
        }
        keys
    }
}
