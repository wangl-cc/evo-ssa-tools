use crate::{
    cache::codec::ValueFormat,
    identity::{
        ComputationPath, FIELD_DISPLAY_SEPARATOR, IdentifierSegmentChain, SEGMENT_ENCODED_SEPARATOR,
    },
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
        let name = format!(
            "{path}{FIELD_DISPLAY_SEPARATOR}{value_format}{FIELD_DISPLAY_SEPARATOR}{short_hash}"
        );

        Self { name }
    }

    /// Return the storage namespace name.
    pub fn as_str(&self) -> &str {
        &self.name
    }
}

fn namespace_hash(path: &ComputationPath, value_format: ValueFormat) -> blake3::Hash {
    let mut hasher = blake3::Hasher::new();
    path.hash_segments(&mut hasher);
    hasher.update(SEGMENT_ENCODED_SEPARATOR);
    value_format.hash_segments(&mut hasher);
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
                .starts_with("computation-a-v1__bitcode06-v1__")
        );
        assert_eq!(namespace.as_str().rsplit_once("__").unwrap().1.len(), 16);
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
