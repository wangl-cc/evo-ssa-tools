//! Borrowed entry views used by segment and block encoders.
//!
//! A segment encoder only requires stable borrowed key/value slices. Callers
//! may store entries in any representation that implements `EntrySource`.

use std::ops::Range;

/// Borrowed key/value record passed into segment encoders.
#[derive(Clone, Copy, Debug)]
pub(crate) struct EntryRef<'a> {
    key: &'a [u8],
    value: &'a [u8],
}

impl<'a> EntryRef<'a> {
    pub(crate) fn new(key: &'a [u8], value: &'a [u8]) -> Self {
        Self { key, value }
    }

    pub(crate) fn key(self) -> &'a [u8] {
        self.key
    }

    pub(crate) fn value(self) -> &'a [u8] {
        self.value
    }
}

/// Indexed source of sorted segment entries.
pub(crate) trait EntrySource {
    fn len(&self) -> usize;

    fn entry(&self, index: usize) -> EntryRef<'_>;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn first_key(&self) -> &[u8] {
        self.entry(0).key()
    }

    fn last_key(&self) -> &[u8] {
        self.entry(self.len() - 1).key()
    }
}

/// Contiguous view into another `EntrySource`.
pub(crate) struct EntryView<'a, S: EntrySource + ?Sized> {
    source: &'a S,
    range: Range<usize>,
}

impl<'a, S: EntrySource + ?Sized> EntryView<'a, S> {
    pub(crate) fn new(source: &'a S, range: Range<usize>) -> Self {
        debug_assert!(range.start <= range.end);
        debug_assert!(range.end <= source.len());
        Self { source, range }
    }
}

impl<S: EntrySource + ?Sized> EntrySource for EntryView<'_, S> {
    fn len(&self) -> usize {
        self.range.end - self.range.start
    }

    fn entry(&self, index: usize) -> EntryRef<'_> {
        assert!(index < self.len(), "entry index should be in range");
        self.source.entry(self.range.start + index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StaticEntries<'a>(&'a [(&'a [u8], &'a [u8])]);

    impl EntrySource for StaticEntries<'_> {
        fn len(&self) -> usize {
            self.0.len()
        }

        fn entry(&self, index: usize) -> EntryRef<'_> {
            let (key, value) = self.0[index];
            EntryRef::new(key, value)
        }
    }

    mod view {
        use super::*;

        #[test]
        fn maps_indices_into_source_range() {
            let entries = StaticEntries(&[(b"k1", b"v1"), (b"k2", b"v2"), (b"k3", b"v3")]);
            let view = EntryView::new(&entries, 1..3);

            assert_eq!(view.len(), 2);
            assert_eq!(view.first_key(), b"k2");
            assert_eq!(view.last_key(), b"k3");
            assert_eq!(view.entry(0).value(), b"v2");
            assert_eq!(view.entry(1).value(), b"v3");
        }
    }
}
