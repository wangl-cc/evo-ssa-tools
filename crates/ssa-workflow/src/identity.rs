//! Semantic computation identifiers and derived computation paths.

/// Stable, versioned identifier for one semantic computation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ComputationId(&'static str);

impl ComputationId {
    /// Create a computation identifier from a stable static name.
    pub const fn new(name: &'static str) -> Self {
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
    pub(crate) fn root(computation: ComputationId) -> Self {
        Self {
            node: Box::new(ComputationPathNode::Root(computation)),
        }
    }

    pub(crate) fn child(&self, computation: ComputationId) -> Self {
        Self {
            node: Box::new(ComputationPathNode::Child {
                parent: self.clone(),
                computation,
            }),
        }
    }

    /// Return the computation IDs from root to leaf.
    pub fn segments(&self) -> Vec<ComputationId> {
        let mut segments = Vec::new();
        self.push_segments(&mut segments);
        segments
    }

    /// Encode this path as stable length-prefixed bytes.
    pub fn encode_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        for segment in self.segments() {
            append_len_prefixed(&mut bytes, segment.as_str().as_bytes());
        }
        bytes
    }

    /// Render this path for diagnostics.
    pub fn render(&self) -> String {
        self.segments()
            .into_iter()
            .map(ComputationId::as_str)
            .collect::<Vec<_>>()
            .join(" -> ")
    }

    pub(crate) fn namespace_hint(&self) -> String {
        self.segments()
            .into_iter()
            .map(|segment| escape_namespace_part(segment.as_str()))
            .collect::<Vec<_>>()
            .join("__")
    }

    fn push_segments(&self, segments: &mut Vec<ComputationId>) {
        match self.node.as_ref() {
            ComputationPathNode::Root(computation) => segments.push(*computation),
            ComputationPathNode::Child {
                parent,
                computation,
            } => {
                parent.push_segments(segments);
                segments.push(*computation);
            }
        }
    }
}

impl std::fmt::Display for ComputationPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.render())
    }
}

pub(crate) fn append_len_prefixed(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(&(bytes.len() as u64).to_be_bytes());
    out.extend_from_slice(bytes);
}

pub(crate) fn escape_namespace_part(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for byte in value.bytes() {
        match byte {
            b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'-' | b'.' => {
                escaped.push(byte as char);
            }
            other => {
                escaped.push('_');
                escaped.push_str(&format!("{other:02x}"));
                escaped.push('_');
            }
        }
    }
    escaped
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn root_path_renders_and_encodes_one_id() {
        let path = ComputationPath::root(ComputationId::new("trajectory/v1"));

        assert_eq!(path.render(), "trajectory/v1");
        assert_eq!(path.namespace_hint(), "trajectory_2f_v1");
        assert_eq!(path.encode_bytes(), [
            0, 0, 0, 0, 0, 0, 0, 13, b't', b'r', b'a', b'j', b'e', b'c', b't', b'o', b'r', b'y',
            b'/', b'v', b'1'
        ]);
    }

    #[test]
    fn child_path_includes_parent_then_child() {
        let root = ComputationPath::root(ComputationId::new("trajectory/v1"));
        let child = root.child(ComputationId::new("summary/v1"));

        assert_eq!(child.segments(), vec![
            ComputationId::new("trajectory/v1"),
            ComputationId::new("summary/v1")
        ]);
        assert_eq!(child.render(), "trajectory/v1 -> summary/v1");
    }

    #[test]
    fn path_display_matches_rendered_diagnostic_path() {
        let path = ComputationPath::root(ComputationId::new("trajectory/v1"))
            .child(ComputationId::new("summary/v1"));

        assert_eq!(path.to_string(), path.render());
    }

    #[test]
    fn sibling_paths_are_distinct() {
        let root = ComputationPath::root(ComputationId::new("trajectory/v1"));
        let peak = root.child(ComputationId::new("peak/v1"));
        let extinct = root.child(ComputationId::new("extinct/v1"));

        assert_ne!(peak, extinct);
        assert_ne!(peak.encode_bytes(), extinct.encode_bytes());
    }

    #[test]
    fn parent_path_changes_child_path() {
        let first = ComputationPath::root(ComputationId::new("model-a/trajectory/v1"))
            .child(ComputationId::new("summary/v1"));
        let second = ComputationPath::root(ComputationId::new("model-b/trajectory/v1"))
            .child(ComputationId::new("summary/v1"));

        assert_ne!(first, second);
    }
}
