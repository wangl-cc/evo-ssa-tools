use crate::{
    cache::codec::ValueFormat,
    identity::{ComputationPath, IdentifierSegmentChain},
};

/// Physical storage namespace derived from a computation path and value format.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StorageNamespace {
    name: String,
}

impl StorageNamespace {
    /// Create a storage namespace from a computation path and value format.
    pub fn new(path: &ComputationPath, value_format: ValueFormat) -> Self {
        let hash = namespace_hash(path, value_format);
        let short_hash: &str = &hash.to_hex()[..16];
        Self {
            name: format!("{path}--{value_format}--{short_hash}"),
        }
    }

    /// Return the storage namespace name.
    pub fn as_str(&self) -> &str {
        &self.name
    }
}

fn namespace_hash(path: &ComputationPath, value_format: ValueFormat) -> blake3::Hash {
    let mut hasher = blake3::Hasher::new();
    path.for_each_segment(|segment| {
        hasher.update(segment.as_bytes());
        hasher.update(&[0]); // Separator
    });
    hasher.update(&[0]); // Separator
    value_format.for_each_segment(|segment| {
        hasher.update(segment.as_bytes());
        hasher.update(&[0]); // Separator
    });
    hasher.finalize()
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
    use crate::identity::ComputationPath;

    const COMPUTATION_A: &str = "computation-a-v1";
    const FORMAT: ValueFormat = ValueFormat::new("bitcode06-v1");

    #[test]
    fn storage_namespace_is_readable_and_includes_value_format() {
        let path = ComputationPath::root_from_str(COMPUTATION_A);
        let namespace = StorageNamespace::new(&path, FORMAT);

        assert!(
            namespace
                .as_str()
                .starts_with("computation-a-v1--bitcode06-v1--")
        );
        assert_eq!(namespace.as_str().rsplit_once("--").unwrap().1.len(), 16);
    }

    #[test]
    fn storage_namespace_display_matches_backend_name() {
        let path = ComputationPath::root_from_str(COMPUTATION_A);
        let namespace = StorageNamespace::new(&path, FORMAT);

        assert_eq!(namespace.to_string(), namespace.as_str());
    }

    #[test]
    fn namespace_hash_includes_path_boundaries() {
        let first = ComputationPath::root_from_str("a-b").child_from_str("c");
        let second = ComputationPath::root_from_str("a").child_from_str("b-c");

        assert_ne!(
            StorageNamespace::new(&first, FORMAT),
            StorageNamespace::new(&second, FORMAT)
        );
    }
}
