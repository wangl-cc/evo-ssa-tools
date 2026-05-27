use crate::{
    cache::codec::ValueFormat,
    identity::{ComputationPath, append_len_prefixed, escape_namespace_part},
};

/// Physical storage namespace derived from a computation path and value format.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StorageNamespace {
    name: String,
}

impl StorageNamespace {
    /// Create a storage namespace from a computation path and value format.
    pub fn new(path: &ComputationPath, value_format: ValueFormat) -> Self {
        let value_format = value_format.render();
        let hash = namespace_hash(path, &value_format);
        Self {
            name: format!(
                "ssa_workflow__path-{}__format-{}__id-{}",
                path.namespace_hint(),
                escape_namespace_part(&value_format),
                hash
            ),
        }
    }

    /// Return the backend namespace name.
    pub fn as_str(&self) -> &str {
        &self.name
    }
}

fn namespace_hash(path: &ComputationPath, value_format: &str) -> String {
    let mut material = Vec::new();
    append_len_prefixed(&mut material, &path.encode_bytes());
    append_len_prefixed(&mut material, value_format.as_bytes());
    blake3::hash(&material).to_hex().to_string()
}

impl std::fmt::Display for StorageNamespace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.name)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::identity::{ComputationId, ComputationPath};

    const COMPUTATION_A: ComputationId = ComputationId::new("computation/a/v1");
    const FORMAT: ValueFormat = ValueFormat::new("bitcode06/v1");

    #[test]
    fn storage_namespace_is_readable_and_includes_value_format() {
        let path = ComputationPath::root(COMPUTATION_A);
        let namespace = StorageNamespace::new(&path, FORMAT);

        assert!(namespace.as_str().contains("path-computation_2f_a_2f_v1"));
        assert!(namespace.as_str().contains("format-bitcode06_2f_v1"));
        assert!(namespace.as_str().contains("__id-"));
    }

    #[test]
    fn storage_namespace_display_matches_backend_name() {
        let path = ComputationPath::root(COMPUTATION_A);
        let namespace = StorageNamespace::new(&path, FORMAT);

        assert_eq!(namespace.to_string(), namespace.as_str());
    }

    #[test]
    fn namespace_escape_is_injective_for_underscore_sequences() {
        let slash = ComputationPath::root(ComputationId::new("a/b"));
        let literal = ComputationPath::root(ComputationId::new("a_2f_b"));

        assert_ne!(
            StorageNamespace::new(&slash, FORMAT),
            StorageNamespace::new(&literal, FORMAT)
        );
    }

    #[test]
    fn namespace_hash_includes_path_boundaries() {
        let first = ComputationPath::root(ComputationId::new("a-b")).child(ComputationId::new("c"));
        let second =
            ComputationPath::root(ComputationId::new("a")).child(ComputationId::new("b-c"));

        assert_ne!(first.encode_bytes(), second.encode_bytes());
        assert_ne!(
            StorageNamespace::new(&first, FORMAT),
            StorageNamespace::new(&second, FORMAT)
        );
    }
}
