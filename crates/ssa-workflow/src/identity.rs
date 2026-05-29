//! Semantic computation identifiers and derived computation paths.

/// Stable, versioned identifier for one semantic computation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ComputationId(&'static str);

impl ComputationId {
    /// Create a computation identifier from a stable static name.
    ///
    /// Names must be non-empty backend-safe identifier segments: ASCII letters, digits, `.`, and
    /// `-` are allowed, but `--` is reserved as a namespace separator.
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

pub(crate) trait IdentifierSegmentChain {
    fn for_each_segment(&self, visit: impl FnMut(&str));

    fn encode_segments(&self) -> Vec<u8>
    where
        Self: Sized,
    {
        let mut bytes = Vec::new();
        self.for_each_segment(|segment| append_len_prefixed(&mut bytes, segment.as_bytes()));
        bytes
    }

    fn render_segments(&self, separator: &str) -> String
    where
        Self: Sized,
    {
        let mut rendered = String::new();
        self.write_segments(separator, &mut rendered)
            .expect("writing to a String should not fail");
        rendered
    }

    fn write_segments<W: std::fmt::Write>(&self, separator: &str, out: &mut W) -> std::fmt::Result
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
                result = out.write_str(separator);
            }
            if result.is_ok() {
                result = out.write_str(segment);
            }
            is_first = false;
        });
        result
    }
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

    /// Render this path for diagnostics.
    pub fn render(&self) -> String {
        self.render_segments(" <- ")
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

impl std::fmt::Display for ComputationPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.write_segments(" <- ", f)
    }
}

// TODO: better way to solve this append prefix (avoiding allocations in some call sites)
pub(crate) fn append_len_prefixed(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(&(bytes.len() as u64).to_be_bytes());
    out.extend_from_slice(bytes);
}

pub(crate) const fn assert_identifier_segment(value: &str, allow_empty: bool) {
    let bytes = value.as_bytes();
    if !allow_empty && bytes.is_empty() {
        panic!("identifier segment must not be empty");
    }

    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        match byte {
            b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'.' | b'-' => {}
            _ => panic!("identifier segment contains an invalid character"),
        }

        if byte == b'-' && index + 1 < bytes.len() && bytes[index + 1] == b'-' {
            panic!("identifier segment must not contain `--`");
        }

        index += 1;
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn root_path_renders_and_encodes_one_id() {
        let path = ComputationPath::root_from_str("trajectory-v1");

        assert_eq!(path.render(), "trajectory-v1");
        assert_eq!(path.render_segments("--"), "trajectory-v1");
        assert_eq!(path.encode_segments(), [
            0, 0, 0, 0, 0, 0, 0, 13, b't', b'r', b'a', b'j', b'e', b'c', b't', b'o', b'r', b'y',
            b'-', b'v', b'1'
        ]);
    }

    #[test]
    fn child_path_iterates_current_then_parent() {
        let root = ComputationPath::root_from_str("trajectory-v1");
        let child = root.child_from_str("summary-v1");

        assert_eq!(child.render_segments("--"), "summary-v1--trajectory-v1");
        assert_eq!(child.render(), "summary-v1 <- trajectory-v1");
    }

    #[test]
    fn path_display_matches_rendered_diagnostic_path() {
        let path = ComputationPath::root_from_str("trajectory-v1").child_from_str("summary-v1");

        assert_eq!(path.to_string(), path.render());
    }

    #[test]
    fn sibling_paths_are_distinct() {
        let root = ComputationPath::root_from_str("trajectory-v1");
        let peak = root.child_from_str("peak-v1");
        let extinct = root.child_from_str("extinct-v1");

        assert_ne!(peak, extinct);
        assert_ne!(peak.encode_segments(), extinct.encode_segments());
    }

    #[test]
    fn parent_path_changes_child_path() {
        let first =
            ComputationPath::root_from_str("model-a-trajectory-v1").child_from_str("summary-v1");
        let second =
            ComputationPath::root_from_str("model-b-trajectory-v1").child_from_str("summary-v1");

        assert_ne!(first, second);
    }

    #[test]
    #[should_panic(expected = "identifier segment contains an invalid character")]
    fn computation_id_rejects_slash() {
        let _ = ComputationId::new("trajectory/v1");
    }

    #[test]
    #[should_panic(expected = "identifier segment must not contain `--`")]
    fn computation_id_rejects_double_hyphen() {
        let _ = ComputationId::new("model--family-v1");
    }

    #[test]
    #[should_panic(expected = "identifier segment contains an invalid character")]
    fn computation_id_rejects_underscore() {
        let _ = ComputationId::new("model_family-v1");
    }

    #[test]
    #[should_panic(expected = "identifier segment must not be empty")]
    fn computation_id_rejects_empty_name() {
        let _ = ComputationId::new("");
    }
}
