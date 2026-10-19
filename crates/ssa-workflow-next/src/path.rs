use std::fmt;

/// Computation path, rendered from the current stage back to its root.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Path(Vec<&'static str>);

impl Path {
    /// The empty path of request roots (repetition indices).
    pub(crate) fn roots() -> Self {
        Self(Vec::new())
    }

    pub(crate) fn child(&self, id: &'static str) -> Self {
        let mut segments = Vec::with_capacity(self.0.len() + 1);
        segments.push(id);
        segments.extend_from_slice(&self.0);
        Self(segments)
    }

    /// Seed material for every random stream of this stage.
    pub(crate) fn seed(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new_derive_key("ssa-workflow-next seed v1");
        for segment in &self.0 {
            hasher.update(segment.as_bytes()).update(b"\0");
        }
        hasher.finalize().into()
    }

    /// Byte prefix of this stage's namespace in the store.
    pub(crate) fn namespace(&self) -> [u8; 16] {
        let mut hasher = blake3::Hasher::new_derive_key("ssa-workflow-next namespace v1");
        for segment in &self.0 {
            hasher.update(segment.as_bytes()).update(b"\0");
        }
        let digest: [u8; 32] = hasher.finalize().into();
        let mut prefix = [0; 16];
        prefix.copy_from_slice(&digest[..16]);
        prefix
    }
}

impl fmt::Display for Path {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            return f.write_str("<roots>");
        }
        f.write_str(&self.0.join("_"))
    }
}
