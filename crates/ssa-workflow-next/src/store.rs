//! The store interface the workflow needs, plus an in-memory implementation.

use std::{
    collections::BTreeMap,
    sync::atomic::{AtomicUsize, Ordering},
};

use parking_lot::RwLock;

use crate::Result;

/// Subset of the segment store (PR #68) used by the workflow. Keys passed to the ordered calls
/// are sorted ascending; namespaces are byte prefixes.
pub trait Store: Sync {
    fn contains_many_ordered(&self, keys: &[&[u8]]) -> Vec<bool>;
    fn visit_many_ordered(&self, keys: &[&[u8]], visitor: &mut dyn FnMut(usize, Option<&[u8]>));
    /// Insert-only: an existing value wins.
    fn put(&self, key: &[u8], value: Vec<u8>);
    /// Durability barrier.
    fn flush(&self) -> Result<()>;
}

/// In-memory store with call counters for tests.
#[derive(Debug, Default)]
pub struct MemoryStore {
    map: RwLock<BTreeMap<Vec<u8>, Vec<u8>>>,
    pub contains_calls: AtomicUsize,
    pub visit_calls: AtomicUsize,
    pub puts: AtomicUsize,
}

impl MemoryStore {
    pub fn len(&self) -> usize {
        self.map.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.read().is_empty()
    }
}

fn assert_sorted(keys: &[&[u8]]) {
    debug_assert!(
        keys.windows(2).all(|pair| pair[0] <= pair[1]),
        "ordered lookup keys must be sorted"
    );
}

impl Store for MemoryStore {
    fn contains_many_ordered(&self, keys: &[&[u8]]) -> Vec<bool> {
        assert_sorted(keys);
        self.contains_calls.fetch_add(1, Ordering::Relaxed);
        let map = self.map.read();
        keys.iter().map(|key| map.contains_key(*key)).collect()
    }

    fn visit_many_ordered(&self, keys: &[&[u8]], visitor: &mut dyn FnMut(usize, Option<&[u8]>)) {
        assert_sorted(keys);
        self.visit_calls.fetch_add(1, Ordering::Relaxed);
        let map = self.map.read();
        for (index, key) in keys.iter().enumerate() {
            visitor(index, map.get(*key).map(Vec::as_slice));
        }
    }

    fn put(&self, key: &[u8], value: Vec<u8>) {
        self.puts.fetch_add(1, Ordering::Relaxed);
        self.map.write().entry(key.to_vec()).or_insert(value);
    }

    fn flush(&self) -> Result<()> {
        Ok(())
    }
}
