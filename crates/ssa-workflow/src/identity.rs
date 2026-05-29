//! Semantic computation identifiers and derived computation paths.

/// Stable, versioned identifier for one semantic computation.
///
/// # Naming convention
///
/// Only alphabetic, digits, and `-` are allowed. `_` is reserved for derived names and must not
/// appear inside one segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ComputationId(&'static str);

impl ComputationId {
    /// Create a computation identifier from a stable static name.
    ///
    /// # Panics
    ///
    /// Panics if the name does not follow the naming convention.
    pub const fn new(name: &'static str) -> Self {
        assert_identifier_segment(name, false);
        Self(name)
    }

    /// Return the computation identifier.
    pub const fn as_str(self) -> &'static str {
        self.0
    }
}

impl From<&'static str> for ComputationId {
    fn from(value: &'static str) -> Self {
        Self::new(value)
    }
}

impl std::fmt::Display for ComputationId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

/// Parent-linked semantic result path for a built compute node.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ComputationPath {
    node: Box<ComputationPathNode>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum ComputationPathNode {
    Root(ComputationId),
    Child {
        parent: ComputationPath,
        computation: ComputationId,
    },
}

impl ComputationPath {
    pub(crate) fn root(id: ComputationId) -> Self {
        Self {
            node: Box::new(ComputationPathNode::Root(id)),
        }
    }

    #[cfg(test)]
    pub(crate) fn root_from_str(id: &'static str) -> Self {
        Self::root(ComputationId::new(id))
    }

    pub(crate) fn child(&self, id: ComputationId) -> Self {
        Self {
            node: Box::new(ComputationPathNode::Child {
                parent: self.clone(),
                computation: id,
            }),
        }
    }

    #[cfg(test)]
    pub(crate) fn child_from_str(&self, id: &'static str) -> Self {
        self.child(ComputationId::new(id))
    }
}

impl std::fmt::Display for ComputationPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.write_segments(f)
    }
}

pub(crate) trait IdentifierSegmentChain {
    fn for_each_segment(&self, visit: impl FnMut(&str));

    fn write_segments<W: std::fmt::Write>(&self, out: &mut W) -> std::fmt::Result
    where
        Self: Sized,
    {
        let mut is_first = true;
        let mut result = Ok(());
        self.for_each_segment(|segment| {
            if result.is_err() {
                return;
            }
            if !is_first {
                result = out.write_str(SEGMENT_DISPLAY_SEPARATOR);
            }
            if result.is_ok() {
                result = out.write_str(segment);
            }
            is_first = false;
        });
        result
    }

    fn hash_segments(&self, hasher: &mut blake3::Hasher) {
        self.for_each_segment(|segment| {
            hasher.update(segment.as_bytes());
            hasher.update(SEGMENT_ENCODED_SEPARATOR);
        });
    }
}

impl IdentifierSegmentChain for ComputationPath {
    fn for_each_segment(&self, mut visit: impl FnMut(&str)) {
        let mut next = Some(self);
        while let Some(path) = next {
            match path.node.as_ref() {
                ComputationPathNode::Root(computation) => {
                    visit(computation.as_str());
                    next = None;
                }
                ComputationPathNode::Child {
                    parent,
                    computation,
                } => {
                    visit(computation.as_str());
                    next = Some(parent);
                }
            }
        }
    }
}

pub(crate) const SEGMENT_DISPLAY_SEPARATOR: &str = "_";

pub(crate) const FIELD_DISPLAY_SEPARATOR: &str = "__";

pub(crate) const SEGMENT_ENCODED_SEPARATOR: &[u8] = b"\0";

pub(crate) const fn assert_identifier_segment(value: &str, allow_empty: bool) {
    let bytes = value.as_bytes();
    if !allow_empty && bytes.is_empty() {
        panic!("identifier segment must not be empty");
    }

    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        match byte {
            b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'-' => {}
            _ => panic!("identifier segment contains an invalid character"),
        }

        index += 1;
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    mod id {
        use super::*;

        #[test]
        fn accepts_number_alphabet_single_hyphen() {
            let _ = ComputationId::new("model-family-v1");
        }

        #[test]
        #[should_panic(expected = "identifier segment contains an invalid character")]
        fn rejects_slash() {
            let _ = ComputationId::new("trajectory/v1");
        }

        #[test]
        fn accepts_double_hyphen() {
            let _ = ComputationId::new("model--family-v1");
        }

        #[test]
        #[should_panic(expected = "identifier segment contains an invalid character")]
        fn rejects_underscore() {
            let _ = ComputationId::new("model_family-v1");
        }

        #[test]
        #[should_panic(expected = "identifier segment must not be empty")]
        fn rejects_empty_name() {
            let _ = ComputationId::new("");
        }
    }

    mod path {
        use super::*;

        #[test]
        fn parent_path_changes_child_path() {
            let first = ComputationPath::root_from_str("model-a-trajectory-v1")
                .child_from_str("summary-v1");
            let second = ComputationPath::root_from_str("model-b-trajectory-v1")
                .child_from_str("summary-v1");

            assert_ne!(first, second);
        }

        #[test]
        fn display() {
            let path = ComputationPath::root_from_str("trajectory-v1").child_from_str("summary-v1");

            assert_eq!(path.to_string(), "summary-v1_trajectory-v1");
        }
    }
}
